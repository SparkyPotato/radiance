use std::collections::hash_map::Entry;

use ash::vk;
use rustc_hash::FxHashMap;

use crate::{
	device::Device,
	graph::FRAMES_IN_FLIGHT,
	resource::{Named, Resource, ToNamed},
	Result,
};

const DESTROY_LAG: u8 = FRAMES_IN_FLIGHT as _;

/// A resource that has its usage generations tracked.
struct TrackedResource<T: Resource> {
	inner: T,
	/// Number of generations the resource has been unused for.
	unused: u8,
}

/// A list of resources with the same descriptor.
pub struct ResourceList<T: Resource> {
	cursor: usize,
	// Stored in most recently used order.
	resources: Vec<TrackedResource<T>>,
}

impl<T: Resource> ResourceList<T> {
	pub fn new() -> Self {
		Self {
			cursor: 0,
			resources: Vec::new(),
		}
	}

	pub fn get_or_create(&mut self, device: &Device, desc: T::Desc<'_>) -> Result<T::Handle> {
		let ret = match self.resources.get_mut(self.cursor) {
			Some(resource) => {
				resource.unused = 0;
				resource.inner.handle()
			},
			None => {
				let resource = T::create(device, desc)?;
				let handle = resource.handle();
				self.resources.push(TrackedResource {
					inner: resource,
					unused: 0,
				});
				handle
			},
		};
		self.cursor += 1;
		Ok(ret)
	}

	pub fn get_all_used(&self) -> impl Iterator<Item = T::Handle> + '_ {
		// Everything before the cursor was just used.
		self.resources[..self.cursor]
			.iter()
			.map(|resource| resource.inner.handle())
	}

	pub unsafe fn reset(&mut self, device: &Device) {
		// Everything before the cursor was just used.
		let mut first_destroyable = self.cursor;

		for resource in self.resources[self.cursor..].iter_mut() {
			// Everything after this has not been used for at least `DESTROY_LAG` generations.
			resource.unused += 1;
			if resource.unused >= DESTROY_LAG {
				break;
			}
			first_destroyable += 1;
		}
		for resource in self.resources.drain(first_destroyable..) {
			resource.inner.destroy(device);
		}
		self.cursor = 0;
	}

	pub fn destroy(self, device: &Device) {
		for resource in self.resources {
			unsafe {
				resource.inner.destroy(device);
			}
		}
	}
}

pub struct ResourceCache<T: Resource> {
	resources: FxHashMap<T::UnnamedDesc, ResourceList<T>>,
}

impl<T: Resource> ResourceCache<T> {
	/// Create an empty cache.
	pub fn new() -> Self {
		Self {
			resources: FxHashMap::default(),
		}
	}

	/// Reset the cache, incrementing the generation.
	///
	/// # Safety
	/// All resources returned by [`Self::get`] must not be used after this call.
	pub unsafe fn reset(&mut self, device: &Device) {
		for (_, list) in self.resources.iter_mut() {
			list.reset(device);
		}
	}

	/// Get an unused resource with the given descriptor. Is valid until [`Self::reset`] is called.
	pub fn get(&mut self, device: &Device, desc: T::UnnamedDesc) -> Result<T::Handle> {
		let list = self.resources.entry(desc).or_insert_with(ResourceList::new);
		list.get_or_create(device, desc.to_named("Graph Resource"))
	}

	pub unsafe fn destroy(self, device: &Device) {
		for (_, list) in self.resources {
			list.destroy(device);
		}
	}
}

pub struct UniqueCache<T: Resource> {
	resources: FxHashMap<T::UnnamedDesc, TrackedResource<T>>,
}

impl<T: Resource> UniqueCache<T> {
	/// Create an empty cache.
	pub fn new() -> Self {
		Self {
			resources: FxHashMap::default(),
		}
	}

	/// Get the resource with the given descriptor. Is valid until [`Self::reset`] is called.
	pub fn get(&mut self, device: &Device, desc: T::UnnamedDesc) -> Result<T::Handle> {
		match self.resources.entry(desc) {
			Entry::Vacant(v) => {
				let resource = T::create(device, desc.to_named("Graph Resource"))?;
				let handle = resource.handle();
				v.insert(TrackedResource {
					inner: resource,
					unused: 0,
				});
				Ok(handle)
			},
			Entry::Occupied(mut o) => {
				let o = o.get_mut();
				o.unused = 0;
				Ok(o.inner.handle())
			},
		}
	}

	/// Reset the cache, incrementing the generation.
	///
	/// # Safety
	/// All resources returned by [`Self::get`] must not be used after this call.
	pub unsafe fn reset(&mut self, device: &Device) {
		self.resources.retain(|_, res| {
			res.unused += 1;
			if res.unused >= DESTROY_LAG {
				std::mem::take(&mut res.inner).destroy(device);
				false
			} else {
				true
			}
		})
	}

	pub fn destroy(self, device: &Device) {
		for (_, res) in self.resources {
			unsafe {
				res.inner.destroy(device);
			}
		}
	}
}

pub struct PersistentCache<T: Resource> {
	resources: FxHashMap<&'static str, (TrackedResource<T>, T::Desc<'static>, vk::ImageLayout)>,
}

impl<T: Resource> PersistentCache<T> {
	/// Create an empty cache.
	pub fn new() -> Self {
		Self {
			resources: FxHashMap::default(),
		}
	}

	/// Get the resource with the given descriptor. Is valid until [`Self::reset`] is called.
	pub fn get(
		&mut self, device: &Device, desc: T::Desc<'static>, next_layout: vk::ImageLayout,
	) -> Result<(T::Handle, vk::ImageLayout)> {
		match self.resources.entry(desc.name()) {
			Entry::Vacant(v) => {
				let resource = T::create(device, desc)?;
				let handle = resource.handle();
				v.insert((
					TrackedResource {
						inner: resource,
						unused: 0,
					},
					desc,
					next_layout,
				));
				Ok((handle, vk::ImageLayout::UNDEFINED))
			},
			Entry::Occupied(mut o) => {
				let (o, d, l) = o.get_mut();
				if *d == desc {
					o.unused = 0;
					let old = *l;
					*l = next_layout;
					Ok((o.inner.handle(), old))
				} else {
					let resource = T::create(device, desc)?;
					let handle = resource.handle();
					let old = std::mem::replace(
						o,
						TrackedResource {
							inner: resource,
							unused: 0,
						},
					);
					unsafe {
						old.inner.destroy(device);
					}
					*d = desc;
					*l = next_layout;
					Ok((handle, vk::ImageLayout::UNDEFINED))
				}
			},
		}
	}

	/// Reset the cache, incrementing the generation.
	///
	/// # Safety
	/// All resources returned by [`Self::get`] must not be used after this call.
	pub unsafe fn reset(&mut self, device: &Device) {
		self.resources.retain(|_, (res, ..)| {
			res.unused += 1;
			if res.unused >= DESTROY_LAG {
				std::mem::take(&mut res.inner).destroy(device);
				false
			} else {
				true
			}
		})
	}

	pub fn destroy(self, device: &Device) {
		for (_, (res, ..)) in self.resources {
			unsafe {
				res.inner.destroy(device);
			}
		}
	}
}
