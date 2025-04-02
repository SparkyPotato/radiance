use std::{
	collections::hash_map::Entry,
	fmt::Debug,
	hash::Hash,
	marker::PhantomData,
	num::NonZeroU64,
	sync::atomic::{AtomicU64, Ordering},
};

use ash::vk;
use rustc_hash::FxHashMap;

use crate::{
	device::Device,
	graph::{VirtualResource, FRAMES_IN_FLIGHT},
	resource::{Resource, ToNamed},
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

	pub fn get_or_create(&mut self, device: &Device, desc: T::Desc<'_>) -> Result<(T::Handle, bool)> {
		let ret = match self.resources.get_mut(self.cursor) {
			Some(resource) => {
				resource.unused = 0;
				(resource.inner.handle(), false)
			},
			None => {
				let resource = T::create(device, desc)?;
				let handle = resource.handle();
				self.resources.push(TrackedResource {
					inner: resource,
					unused: 0,
				});
				(handle, true)
			},
		};
		self.cursor += 1;
		Ok(ret)
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
	pub fn get(&mut self, device: &Device, desc: T::UnnamedDesc) -> Result<(T::Handle, bool)> {
		let list = self.resources.entry(desc).or_insert_with(ResourceList::new);
		list.get_or_create(device, desc.to_named("unnamed graph resource"))
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
	pub fn get(&mut self, device: &Device, desc: T::UnnamedDesc) -> Result<(T::Handle, bool)> {
		match self.resources.entry(desc) {
			Entry::Vacant(v) => {
				let resource = T::create(device, desc.to_named("unnamed graph resource"))?;
				let handle = resource.handle();
				v.insert(TrackedResource {
					inner: resource,
					unused: 0,
				});
				Ok((handle, true))
			},
			Entry::Occupied(mut o) => {
				let o = o.get_mut();
				o.unused = 0;
				Ok((o.inner.handle(), false))
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

struct PersistentResource<T: Resource> {
	resource: TrackedResource<T>,
	desc: T::UnnamedDesc,
	age: u64,
	layout: vk::ImageLayout,
}

pub struct Persist<T: VirtualResource> {
	pub(crate) key: NonZeroU64,
	_phantom: PhantomData<fn() -> T>,
}
impl<T: VirtualResource> Copy for Persist<T> {}
impl<T: VirtualResource> Clone for Persist<T> {
	fn clone(&self) -> Self { *self }
}
impl<T: VirtualResource> Hash for Persist<T> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.key.hash(state) }
}
impl<T: VirtualResource> PartialEq for Persist<T> {
	fn eq(&self, other: &Self) -> bool { self.key == other.key }
}
impl<T: VirtualResource> Eq for Persist<T> {}
impl<T: VirtualResource> Debug for Persist<T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Persist({})", self.key.get()) }
}

static COUNTER: AtomicU64 = AtomicU64::new(1);
impl<T: VirtualResource> Persist<T> {
	pub fn new() -> Self {
		Self {
			key: NonZeroU64::new(COUNTER.fetch_add(1, Ordering::Relaxed)).unwrap(),
			_phantom: PhantomData,
		}
	}
}

pub struct PersistentCache<T: Resource> {
	resources: FxHashMap<NonZeroU64, PersistentResource<T>>,
}

impl<T: Resource> PersistentCache<T> {
	/// Create an empty cache.
	pub fn new() -> Self {
		Self {
			resources: FxHashMap::default(),
		}
	}

	pub fn get_desc(&self, key: NonZeroU64) -> Option<T::UnnamedDesc> { self.resources.get(&key).map(|r| r.desc) }

	/// Get the resource with the given descriptor. Is valid until [`Self::reset`] is called.
	pub fn get(
		&mut self, device: &Device, key: NonZeroU64, desc: T::UnnamedDesc, next_layout: vk::ImageLayout,
	) -> Result<(T::Handle, bool, vk::ImageLayout)> {
		match self.resources.entry(key) {
			Entry::Vacant(v) => {
				let resource = T::create(device, desc.to_named("graph resource"))?;
				let handle = resource.handle();
				v.insert(PersistentResource {
					resource: TrackedResource {
						inner: resource,
						unused: 0,
					},
					desc,
					age: 0,
					layout: next_layout,
				});
				Ok((handle, true, vk::ImageLayout::UNDEFINED))
			},
			Entry::Occupied(mut o) => {
				let r = o.get_mut();
				if r.desc == desc {
					r.resource.unused = 0;
					let old = r.layout;
					r.layout = next_layout;
					r.age += 1;
					Ok((r.resource.inner.handle(), r.age < 1, old))
				} else {
					let resource = T::create(device, desc.to_named("graph resource"))?;
					let handle = resource.handle();
					let old = std::mem::replace(
						&mut r.resource,
						TrackedResource {
							inner: resource,
							unused: 0,
						},
					);
					unsafe {
						old.inner.destroy(device);
					}
					r.age = 0;
					r.desc = desc;
					r.layout = next_layout;
					Ok((handle, true, vk::ImageLayout::UNDEFINED))
				}
			},
		}
	}

	/// Reset the cache, incrementing the generation.
	///
	/// # Safety
	/// All resources returned by [`Self::get`] must not be used after this call.
	pub unsafe fn reset(&mut self, device: &Device) {
		self.resources.retain(|_, r| {
			r.resource.unused += 1;
			if r.resource.unused >= DESTROY_LAG {
				std::mem::take(&mut r.resource.inner).destroy(device);
				false
			} else {
				true
			}
		})
	}

	pub fn destroy(self, device: &Device) {
		for (_, r) in self.resources {
			unsafe {
				r.resource.inner.destroy(device);
			}
		}
	}
}
