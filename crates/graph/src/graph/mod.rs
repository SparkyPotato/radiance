//! The render graph.

use std::{
	alloc::{Allocator, Layout},
	hash::BuildHasherDefault,
	marker::PhantomData,
};

use ash::vk;
use hashbrown::HashMap;
use rustc_hash::FxHasher;
use tracing::{span, Level};

pub use crate::graph::{
	frame_data::{Deletable, Resource},
	virtual_resource::{
		BufferDesc,
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
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{
		cache::{ResourceCache, ResourceList, UniqueCache},
		compile::{CompiledFrame, DataState, ResourceMap},
		frame_data::{FrameData, Submitter},
		virtual_resource::{ResourceLifetime, VirtualResourceData},
	},
	resource::{Buffer, Event, Image, ImageView},
	Result,
};

mod cache;
mod compile;
mod frame_data;
pub mod util;
mod virtual_resource;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// The render graph.
pub struct RenderGraph {
	frame_data: [FrameData; FRAMES_IN_FLIGHT],
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

pub struct Caches {
	pub upload_buffers: [ResourceCache<Buffer>; FRAMES_IN_FLIGHT],
	pub buffers: ResourceCache<Buffer>,
	pub images: ResourceCache<Image>,
	pub image_views: UniqueCache<ImageView>,
	pub events: ResourceList<Event>,
}

impl RenderGraph {
	pub fn new<'a>(device: &Device) -> Result<Self> {
		let frame_data = [FrameData::new(device)?, FrameData::new(device)?];

		let caches = Caches {
			upload_buffers: [ResourceCache::new(), ResourceCache::new()],
			buffers: ResourceCache::new(),
			images: ResourceCache::new(),
			image_views: UniqueCache::new(),
			events: ResourceList::new(),
		};

		Ok(Self {
			frame_data,
			caches,
			curr_frame: 0,
			resource_base_id: 0,
		})
	}

	pub fn frame<'pass, 'graph>(&'graph mut self, device: &'graph Device) -> Frame<'pass, 'graph> {
		Frame {
			graph: self,
			device,
			passes: Vec::new_in(device.arena()),
			virtual_resources: Vec::new_in(device.arena()),
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			let _ = device.device().device_wait_idle();
			for frame_data in self.frame_data {
				frame_data.destroy(device);
			}
			for cache in self.caches.upload_buffers {
				cache.destroy(device);
			}
			self.caches.buffers.destroy(device);
			self.caches.image_views.destroy(device);
			self.caches.images.destroy(device);
			self.caches.events.destroy(device);
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

	pub fn device(&self) -> &'graph Device { &self.device }

	pub fn arena(&self) -> &'graph Arena { self.device.arena() }

	pub fn start_region(&mut self, name: &str) {
		let name = name.as_bytes().iter().copied().chain([0]);
		self.passes
			.push(FrameEvent::RegionStart(name.collect_in(self.device.arena())));
	}

	pub fn end_region(&mut self) { self.passes.push(FrameEvent::RegionEnd); }

	/// Build a pass with a name.
	pub fn pass(&mut self, name: &str) -> PassBuilder<'_, 'pass, 'graph> {
		self.start_region(name);
		PassBuilder { frame: self }
	}
}

impl Frame<'_, '_> {
	pub fn delete(&mut self, res: impl Deletable) { self.graph.frame_data[self.graph.curr_frame].delete(res); }

	/// Run the frame.
	pub fn run(self) -> Result<()> {
		let device = self.device;
		let arena = device.arena();
		let data = &mut self.graph.frame_data[self.graph.curr_frame];
		data.reset(device)?;
		unsafe {
			self.graph.caches.upload_buffers[self.graph.curr_frame].reset(device);
			self.graph.caches.buffers.reset(device);
			self.graph.caches.image_views.reset(device);
			self.graph.caches.images.reset(device);
			self.graph.caches.events.reset(device);
		}

		let CompiledFrame {
			passes,
			sync,
			mut resource_map,
			graph,
		} = self.compile(device)?;

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
								submitter.pass(device)?,
								&vk::DebugUtilsLabelEXT::default()
									.label_name(std::ffi::CStr::from_bytes_with_nul_unchecked(&name)),
							);
						}
					}
				},
				FrameEvent::RegionEnd => unsafe {
					region_stack.pop();
					if let Some(debug) = device.debug_utils_ext() {
						debug.cmd_end_debug_utils_label(submitter.pass(device)?);
					}
				},
				FrameEvent::Pass(pass) => {
					let buf = submitter.pass(device)?;

					(pass.callback)(PassContext {
						arena,
						device,
						buf,
						base_id: graph.resource_base_id,
						pass: i as u32,
						resource_map: &mut resource_map,
						caches: &mut graph.caches,
					});
				},
			}
		}

		submitter.finish(device, |buf| {
			for event in graph.caches.events.get_all_used() {
				unsafe {
					device
						.device()
						.cmd_reset_event2(buf, event, vk::PipelineStageFlags2::ALL_COMMANDS);
				}
			}
		})?;

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
	pub fn reference<T: VirtualResource>(&mut self, id: Res<T>, usage: T::Usage<'_>) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);

		unsafe {
			let res = self.frame.virtual_resources.get_unchecked_mut(id);
			res.lifetime.end = self.frame.passes.len() as _;
			T::add_read_usage(res, self.frame.passes.len() as _, usage, self.frame.device.arena());
		}
	}

	/// Output GPU data for other passes.
	pub fn resource<D: VirtualResourceDesc>(
		&mut self, desc: D, usage: <D::Resource as VirtualResource>::Usage<'_>,
	) -> Res<D::Resource> {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let ty = desc.ty(
			self.frame.passes.len() as _,
			usage,
			&mut self.frame.virtual_resources,
			self.frame.graph.resource_base_id,
			self.frame.device.arena(),
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

	/// Read CPU data that another pass outputs.
	pub fn data_input<T>(&mut self, id: &GetId<T>) { let _ = id; }

	/// Just like [`Self::data_input`], but the pass only gets a reference to the data.
	pub fn data_input_ref<T>(&mut self, id: RefId<T>) { let _ = id; }

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

	/// Build the pass with the given callback.
	pub fn build(self, callback: impl FnOnce(PassContext<'_, 'graph>) + 'pass) {
		let pass = PassData {
			callback: Box::new_in(callback, self.frame.device.arena()),
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
}

impl<'frame, 'graph> PassContext<'frame, 'graph> {
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

	pub fn get_caches(&mut self) -> &mut Caches { self.caches }
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

type ArenaMap<'graph, K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>, &'graph Arena>;
