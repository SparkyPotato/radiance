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

pub use crate::graph::virtual_resource::{
	BufferDesc,
	BufferUsage,
	BufferUsageType,
	ExternalBuffer,
	ExternalImage,
	ImageDesc,
	ImageUsage,
	ImageUsageType,
	Shader,
	Signal,
	VirtualResource,
	VirtualResourceDesc,
	VirtualResourceType,
	Wait,
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
mod virtual_resource;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// The render graph.
pub struct RenderGraph {
	frame_data: [FrameData; FRAMES_IN_FLIGHT],
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

#[doc(hidden)]
pub struct Caches {
	upload_buffers: [ResourceCache<Buffer>; FRAMES_IN_FLIGHT],
	buffers: ResourceCache<Buffer>,
	images: ResourceCache<Image>,
	image_views: UniqueCache<ImageView>,
	events: ResourceList<Event>,
}

impl RenderGraph {
	pub fn new(device: &Device) -> Result<Self> {
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

	pub fn frame<'pass, 'graph, C>(&'graph mut self, arena: &'graph Arena, ctx: C) -> Frame<'pass, 'graph, C> {
		Frame {
			graph: self,
			arena,
			passes: Vec::new_in(arena),
			virtual_resources: Vec::new_in(arena),
			ctx,
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			for frame_data in self.frame_data {
				frame_data.destroy(device);
			}
			for cache in self.caches.upload_buffers {
				cache.destroy(device);
			}
			self.caches.buffers.destroy(device);
			self.caches.image_views.destroy(device);
			self.caches.images.destroy(device);
		}
	}

	fn next_frame(&mut self, resource_count: usize) {
		self.curr_frame ^= 1;
		self.resource_base_id = self.resource_base_id.wrapping_add(resource_count);
	}
}

/// A frame being recorded to run in the render graph.
pub struct Frame<'pass, 'graph, C> {
	graph: &'graph mut RenderGraph,
	arena: &'graph Arena,
	passes: Vec<PassData<'pass, 'graph, C>, &'graph Arena>,
	virtual_resources: Vec<VirtualResourceData<'graph>, &'graph Arena>,
	ctx: C,
}

impl<'pass, 'graph, C> Frame<'pass, 'graph, C> {
	pub fn graph(&self) -> &RenderGraph { self.graph }

	pub fn arena(&self) -> &'graph Arena { self.arena }

	pub fn ctx(&mut self) -> &mut C { &mut self.ctx }

	/// Build a pass with a name.
	pub fn pass(&mut self, name: &str) -> PassBuilder<'_, 'pass, 'graph, C> {
		let arena = self.arena;
		let name = name.as_bytes().iter().copied().chain([0]);
		PassBuilder {
			name: name.collect_in(arena),
			frame: self,
		}
	}

	/// Run the frame.
	pub fn run(self, device: &Device) -> Result<()> {
		let arena = self.arena;
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
			mut ctx,
		} = self.compile(device)?;

		let span = span!(Level::TRACE, "run passes");
		let _e = span.enter();

		let mut submitter = Submitter::new(arena, sync, &mut graph.frame_data, graph.curr_frame);

		for (i, pass) in passes.into_iter().enumerate() {
			{
				let name = unsafe { std::str::from_utf8_unchecked(&pass.name[..pass.name.len() - 1]) };
				let span = span!(Level::TRACE, "run pass", name = name);
				let _e = span.enter();

				let buf = submitter.pass(device)?;

				#[cfg(debug_assertions)]
				unsafe {
					if let Some(debug) = device.debug_utils_ext() {
						debug.cmd_begin_debug_utils_label(
							buf,
							&vk::DebugUtilsLabelEXT::builder()
								.label_name(std::ffi::CStr::from_bytes_with_nul_unchecked(&pass.name)),
						);
					}
				}

				(pass.callback)(PassContext {
					arena,
					device,
					buf,
					base_id: graph.resource_base_id,
					pass: i as u32,
					resource_map: &mut resource_map,
					caches: &mut graph.caches,
					ctx: &mut ctx,
				});

				#[cfg(debug_assertions)]
				unsafe {
					if let Some(debug) = device.debug_utils_ext() {
						debug.cmd_end_debug_utils_label(buf);
					}
				}
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
pub struct PassBuilder<'frame, 'pass, 'graph, C> {
	name: Vec<u8, &'graph Arena>,
	frame: &'frame mut Frame<'pass, 'graph, C>,
}

impl<'frame, 'pass, 'graph, C> PassBuilder<'frame, 'pass, 'graph, C> {
	/// Get the frame context.
	pub fn ctx(&mut self) -> &mut C { &mut self.frame.ctx }

	/// Read GPU data that another pass outputs.
	pub fn input<T: VirtualResource>(&mut self, id: Res<T>, usage: T::Usage<'_>) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);

		unsafe {
			let res = self.frame.virtual_resources.get_unchecked_mut(id);
			res.lifetime.end = self.frame.passes.len() as _;
			T::add_read_usage(res, self.frame.passes.len() as _, usage);
		}
	}

	/// Output GPU data for other passes.
	pub fn output<D: VirtualResourceDesc>(
		&mut self, desc: D, usage: <D::Resource as VirtualResource>::Usage<'_>,
	) -> Res<D::Resource> {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let ty = desc.ty(
			self.frame.passes.len() as _,
			usage,
			&mut self.frame.virtual_resources,
			self.frame.graph.resource_base_id,
			self.frame.arena,
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
			ty: VirtualResourceType::Data(self.frame.arena.allocate(Layout::new::<T>()).unwrap().cast()),
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

	/// Build the pass with the given callback.
	pub fn build(self, callback: impl FnOnce(PassContext<'_, 'graph, C>) + 'pass) {
		let pass = PassData {
			name: self.name,
			callback: Box::new_in(callback, self.frame.arena),
		};
		self.frame.passes.push(pass);
	}
}

/// Context given to the callback for every pass.
pub struct PassContext<'frame, 'graph, C> {
	pub arena: &'graph Arena,
	pub device: &'frame Device,
	pub buf: vk::CommandBuffer,
	pub ctx: &'frame mut C,
	base_id: usize,
	pass: u32,
	resource_map: &'frame mut ResourceMap<'graph>,
	caches: &'frame mut Caches,
}

impl<'frame, 'graph, C> PassContext<'frame, 'graph, C> {
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

struct PassData<'pass, 'graph, C> {
	// UTF-8 encoded, null terminated.
	name: Vec<u8, &'graph Arena>,
	callback: Box<dyn FnOnce(PassContext<'_, 'graph, C>) + 'pass, &'graph Arena>,
}

type ArenaMap<'graph, K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>, &'graph Arena>;
