//! The render graph.

use std::{
	alloc::{Allocator, Layout},
	hash::BuildHasherDefault,
	marker::PhantomData,
};

use ash::vk;
use hashbrown::HashMap;
use rustc_hash::FxHasher;
use tracing::{Level, span};

pub use crate::graph::{
	cache::Persist,
	virtual_resource::{
		BufferDesc,
		BufferLoc,
		BufferUsage,
		BufferUsageType,
		ExternalBuffer,
		ExternalImage,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		Shader,
		SwapchainImage,
		VirtualResource,
		VirtualResourceDesc,
		VirtualResourceType,
	},
};
use crate::{
	Result,
	arena::{Arena, IteratorAlloc, ToOwnedAlloc},
	device::Device,
	graph::{
		cache::{PersistentCache, ResourceCache, UniqueCache},
		compile::{CompiledFrame, DataState, ResourceMap},
		deleter::{Deletable, Deleter},
		frame_data::{FrameData, Submitter},
		virtual_resource::{ResourceLifetime, VirtualResourceData},
	},
	resource::{Buffer, Image, ImageView},
};

mod cache;
mod compile;
mod deleter;
mod frame_data;
mod virtual_resource;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// The render graph.
pub struct RenderGraph {
	frame_data: [FrameData; FRAMES_IN_FLIGHT],
	deleter: Deleter,
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

pub struct Caches {
	pub upload_buffers: [ResourceCache<Buffer>; FRAMES_IN_FLIGHT],
	pub buffers: ResourceCache<Buffer>,
	pub persistent_buffers: PersistentCache<Buffer>,
	pub readback_buffers: [PersistentCache<Buffer>; FRAMES_IN_FLIGHT],
	pub images: ResourceCache<Image>,
	pub persistent_images: PersistentCache<Image>,
	pub image_views: UniqueCache<ImageView>,
}

impl RenderGraph {
	pub fn new<'a>(device: &Device) -> Result<Self> {
		let frame_data = [FrameData::new(device)?, FrameData::new(device)?];

		let caches = Caches {
			upload_buffers: [ResourceCache::new(), ResourceCache::new()],
			buffers: ResourceCache::new(),
			persistent_buffers: PersistentCache::new(),
			readback_buffers: [PersistentCache::new(), PersistentCache::new()],
			images: ResourceCache::new(),
			persistent_images: PersistentCache::new(),
			image_views: UniqueCache::new(),
		};

		Ok(Self {
			frame_data,
			deleter: Deleter::new(),
			caches,
			curr_frame: 0,
			resource_base_id: 0,
		})
	}

	pub fn frame<'pass, 'graph>(
		&'graph mut self, device: &'graph Device, arena: &'graph Arena,
	) -> Result<Frame<'pass, 'graph>> {
		self.frame_data[self.curr_frame].reset(device)?;
		Ok(Frame {
			graph: self,
			device,
			passes: Vec::new_in(arena),
			virtual_resources: Vec::new_in(arena),
		})
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			let _ = device.device().device_wait_idle();
			self.deleter.destroy(device);
			for frame_data in self.frame_data {
				frame_data.destroy(device);
			}
			for cache in self.caches.upload_buffers {
				cache.destroy(device);
			}
			self.caches.buffers.destroy(device);
			self.caches.persistent_buffers.destroy(device);
			for cache in self.caches.readback_buffers {
				cache.destroy(device);
			}
			self.caches.image_views.destroy(device);
			self.caches.images.destroy(device);
			self.caches.persistent_images.destroy(device);
		}
	}

	fn next_frame(&mut self, resource_count: usize) {
		self.curr_frame ^= 1;
		self.resource_base_id = self.resource_base_id.wrapping_add(resource_count);
	}
}

enum FrameEvent<'pass, 'graph> {
	Pass(PassData<'pass, 'graph>),
	RegionStart(Vec<u8, &'graph Arena>),
	RegionEnd,
}

/// A frame being recorded to run in the render graph.
pub struct Frame<'pass, 'graph> {
	graph: &'graph mut RenderGraph,
	device: &'graph Device,
	passes: Vec<FrameEvent<'pass, 'graph>, &'graph Arena>,
	virtual_resources: Vec<VirtualResourceData<'graph>, &'graph Arena>,
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub fn graph(&self) -> &RenderGraph { self.graph }

	pub fn device(&self) -> &'graph Device { self.device }

	pub fn arena(&self) -> &'graph Arena { self.passes.allocator() }

	pub fn start_region(&mut self, name: &str) {
		let name = name.as_bytes().iter().copied().chain([0]);
		self.passes.push(FrameEvent::RegionStart(name.collect_in(self.arena())));
	}

	pub fn end_region(&mut self) { self.passes.push(FrameEvent::RegionEnd); }

	/// Build a pass with a name.
	pub fn pass(&mut self, name: &str) -> PassBuilder<'_, 'pass, 'graph> {
		self.start_region(name);
		PassBuilder { frame: self }
	}
}

impl Frame<'_, '_> {
	pub fn delete(&mut self, res: impl Deletable) { self.graph.deleter.push(res); }

	/// Run the frame.
	pub fn run(self) -> Result<()> {
		let span = span!(Level::TRACE, "exec frame");
		let _e = span.enter();

		let device = self.device;
		let arena = self.arena();
		// SAFETY: data is reset when the frame is constructed.
		unsafe {
			self.graph.frame_data[self.graph.curr_frame].reset(device)?;
			self.graph.deleter.next(device);
			self.graph.caches.upload_buffers[self.graph.curr_frame].reset(device);
			self.graph.caches.buffers.reset(device);
			self.graph.caches.persistent_buffers.reset(device);
			self.graph.caches.readback_buffers[self.graph.curr_frame].reset(device);
			self.graph.caches.image_views.reset(device);
			self.graph.caches.images.reset(device);
			self.graph.caches.persistent_images.reset(device);
		}

		let CompiledFrame {
			passes,
			sync,
			mut resource_map,
			graph,
		} = self.compile(device, arena)?;

		let span = span!(Level::TRACE, "run passes");
		let _e = span.enter();

		let mut submitter = Submitter::new(arena, sync, &mut graph.frame_data, graph.curr_frame);

		let mut region_stack = Vec::new_in(arena);
		for (i, pass) in passes.into_iter().enumerate() {
			match pass {
				FrameEvent::RegionStart(name) => {
					let span = span!(
						Level::TRACE,
						"graph exec",
						name = unsafe { std::str::from_utf8_unchecked(&name[..name.len() - 1]) }
					);
					region_stack.push(span.entered());

					unsafe {
						if let Some(debug) = device.debug_utils_ext() {
							debug.cmd_begin_debug_utils_label(
								submitter.before_pass(device)?,
								&vk::DebugUtilsLabelEXT::default()
									.label_name(std::ffi::CStr::from_bytes_with_nul_unchecked(&name)),
							);
						}
					}
				},
				FrameEvent::RegionEnd => unsafe {
					region_stack.pop();
					if let Some(debug) = device.debug_utils_ext() {
						debug.cmd_end_debug_utils_label(submitter.before_pass(device)?);
					}
				},
				FrameEvent::Pass(pass) => (pass.callback)(PassContext {
					arena,
					device,
					buf: submitter.before_pass(device)?,
					base_id: graph.resource_base_id,
					pass: i as u32,
					resource_map: &mut resource_map,
					caches: &mut graph.caches,
					deleter: &mut graph.deleter,
				}),
			}
		}

		submitter.finish(device)?;

		let len = resource_map.cleanup();
		graph.next_frame(len);

		Ok(())
	}
}

/// A builder for a pass.
pub struct PassBuilder<'frame, 'pass, 'graph> {
	frame: &'frame mut Frame<'pass, 'graph>,
}

impl<'frame, 'pass, 'graph> PassBuilder<'frame, 'pass, 'graph> {
	/// Read GPU data that another pass outputs.
	pub fn reference<T: VirtualResource>(
		&mut self, id: Res<T>, usage: impl ToOwnedAlloc<Owned<&'graph Arena> = T::Usage<&'graph Arena>>,
	) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);

		unsafe {
			let arena = self.frame.arena();
			let res = self.frame.virtual_resources.get_unchecked_mut(id);
			res.lifetime.end = self.frame.passes.len() as _;
			T::add_read_usage(res, self.frame.passes.len() as _, usage.to_owned_alloc(arena));
		}
	}

	/// Output GPU data for other passes.
	pub fn resource<D: VirtualResourceDesc>(
		&mut self, desc: D,
		usage: impl ToOwnedAlloc<Owned<&'graph Arena> = <D::Resource as VirtualResource>::Usage<&'graph Arena>>,
	) -> Res<D::Resource> {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let arena = self.frame.arena();
		let ty = desc.ty(
			self.frame.passes.len() as _,
			usage.to_owned_alloc(arena),
			&mut self.frame.virtual_resources,
			self.frame.graph.resource_base_id,
		);

		self.frame.virtual_resources.push(VirtualResourceData {
			lifetime: ResourceLifetime::singular(self.frame.passes.len() as _),
			ty,
		});

		Res {
			id,
			_marker: PhantomData,
		}
	}

	/// Output some CPU data for other passes.
	pub fn data_output<T>(&mut self) -> (SetId<T>, GetId<T>) {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.frame.virtual_resources.push(VirtualResourceData {
			lifetime: ResourceLifetime::singular(self.frame.passes.len() as _),
			ty: VirtualResourceType::Data(self.frame.arena().allocate(Layout::new::<T>()).unwrap().cast()),
		});

		(
			SetId {
				id,
				_marker: PhantomData,
			},
			GetId {
				id,
				_marker: PhantomData,
			},
		)
	}

	pub fn desc<T: VirtualResource>(&mut self, res: Res<T>) -> T::Desc {
		let data = &self.frame.virtual_resources[res.id - self.frame.graph.resource_base_id];
		unsafe { T::desc(data) }
	}

	pub fn persistent_desc<T: VirtualResource>(&mut self, res: Persist<T>) -> Option<T::Desc> {
		T::persistent_desc(&self.frame.graph.caches, res)
	}

	/// Build the pass with the given callback.
	pub fn build(self, callback: impl FnOnce(PassContext<'_, 'graph>) + 'pass) {
		let pass = PassData {
			callback: Box::new_in(callback, self.frame.arena()),
		};
		self.frame.passes.push(FrameEvent::Pass(pass));
		self.frame.end_region();
	}
}

/// Context given to the callback for every pass.
pub struct PassContext<'frame, 'graph> {
	pub arena: &'graph Arena,
	pub device: &'frame Device,
	pub buf: vk::CommandBuffer,
	base_id: usize,
	pass: u32,
	resource_map: &'frame mut ResourceMap<'graph>,
	caches: &'frame mut Caches,
	deleter: &'frame mut Deleter,
}

impl<'frame, 'graph> PassContext<'frame, 'graph> {
	pub fn delete(&mut self, res: impl Deletable) { self.deleter.push(res); }

	pub fn desc<T: VirtualResource>(&mut self, res: Res<T>) -> T::Desc {
		let id = res.id.wrapping_sub(self.base_id);
		unsafe {
			let data = self.resource_map.get_virtual(id as u32);
			T::desc(data)
		}
	}

	/// Get a reference to transient CPU-side data output by another pass.
	pub fn get_data_ref<T: 'frame>(&mut self, id: RefId<T>) -> &'frame T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data();

			assert!(
				matches!(state, DataState::Init { .. }),
				"Transient Data has not been initialized"
			);
			&*ptr.as_ptr()
		}
	}

	/// Get owned transient CPU-side data output by another pass.
	pub fn get_data<T: 'frame>(&mut self, id: GetId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data::<T>();

			assert!(
				matches!(state, DataState::Init { .. }),
				"Transient Data has not been initialized"
			);
			let data = ptr.as_ptr().read();
			*state = DataState::Uninit;
			data
		}
	}

	/// Set transient CPU-side data as an output of this pass.
	pub fn set_data<T: 'frame>(&mut self, id: SetId<T>, data: T) {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data::<T>();

			ptr.as_ptr().write(data);
			*state = DataState::Init {
				drop: |ptr| {
					let ptr = ptr.as_ptr() as *mut T;
					ptr.drop_in_place();
				},
			}
		}
	}

	/// Get a GPU resource.
	pub fn get<T: VirtualResource>(&mut self, id: Res<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			T::from_res(self.pass, res, self.caches, self.device)
		}
	}

	/// Is a resource uninitialized? This will be true for everything except persistent resources
	/// that were not created this frame.
	pub fn is_uninit<T: VirtualResource>(&mut self, id: Res<T>) -> bool {
		let id = id.id.wrapping_sub(self.base_id);
		let res = self.resource_map.get(id as u32);
		res.uninit()
	}

	pub fn caches(&mut self) -> &mut Caches { self.caches }
}

/// An ID to write CPU-side data.
pub struct SetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

/// An ID to read CPU-side data.
pub struct GetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T: Copy> Copy for GetId<T> {}
impl<T: Copy> Clone for GetId<T> {
	fn clone(&self) -> Self { *self }
}

impl<T> GetId<T> {
	/// Convert this to a reference ID.
	pub fn to_ref(self) -> RefId<T> {
		RefId {
			id: self.id,
			_marker: PhantomData,
		}
	}
}

/// An ID to read CPU-side data as references.
pub struct RefId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T> Copy for RefId<T> {}
impl<T> Clone for RefId<T> {
	fn clone(&self) -> Self { *self }
}

impl<T> From<GetId<T>> for RefId<T> {
	fn from(id: GetId<T>) -> Self { id.to_ref() }
}

/// An ID to read GPU resources.
pub struct Res<T: VirtualResource> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T: VirtualResource> Res<T> {
	pub fn into_raw(self) -> usize { self.id }

	pub unsafe fn from_raw(id: usize) -> Self {
		Self {
			id,
			_marker: PhantomData,
		}
	}
}

impl<T: VirtualResource> Copy for Res<T> {}
impl<T: VirtualResource> Clone for Res<T> {
	fn clone(&self) -> Self { *self }
}

struct PassData<'pass, 'graph> {
	callback: Box<dyn FnOnce(PassContext<'_, 'graph>) + 'pass, &'graph Arena>,
}

pub type ArenaMap<'graph, K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>, &'graph Arena>;
pub type ArenaSet<'graph, T> = hashbrown::HashSet<T, BuildHasherDefault<FxHasher>, &'graph Arena>;
