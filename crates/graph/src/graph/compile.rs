use std::{alloc::Allocator, hash::BuildHasherDefault, hint::unreachable_unchecked, ops::BitOr, ptr::NonNull};

use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	device::{Device, QueueWaitOwned, SyncStage},
	graph::{
		virtual_resource::{
			compatible_formats,
			BufferData,
			BufferUsageOwned,
			GpuData,
			ImageData,
			ImageUsageOwned,
			ResourceLifetime,
			VirtualResourceData,
			VirtualResourceType,
		},
		ArenaMap,
		BufferLoc,
		Frame,
		FrameEvent,
		ImageDesc,
		RenderGraph,
	},
	resource::{BufferHandle, Subresource},
	sync::{
		as_next_access,
		as_previous_access,
		get_access_info,
		is_write_access,
		AccessInfo,
		GlobalBarrier,
		GlobalBarrierAccess,
		ImageBarrierAccess,
		UsageType,
	},
	Result,
};

pub(super) struct CompiledFrame<'pass, 'graph> {
	pub passes: Vec<FrameEvent<'pass, 'graph>, &'graph Arena>,
	/// First sync is before the first pass, then interspersed between passes, and then the last sync is after the last
	/// pass.
	pub sync: Vec<Sync<'graph>, &'graph Arena>,
	pub resource_map: ResourceMap<'graph>,
	pub graph: &'graph mut RenderGraph,
}

#[derive(Debug)]
pub struct CrossQueueSync<'graph> {
	/// Semaphores to signal. Signals occur before waits.
	pub signal: Vec<SyncStage<vk::Semaphore>, &'graph Arena>,
	/// The synchronization that happens before the signal.
	pub signal_barriers: DependencyInfo<'graph>,
	/// Semaphores to wait on. Waits occur after signals.
	pub wait: QueueWaitOwned<&'graph Arena>,
	/// The synchronization that happens after the wait.
	pub wait_barriers: DependencyInfo<'graph>,
}

#[derive(Clone, Debug)]
pub struct DependencyInfo<'graph> {
	/// Global barriers.
	pub barriers: Vec<vk::MemoryBarrier2<'static>, &'graph Arena>,
	/// Image barriers.
	pub image_barriers: Vec<vk::ImageMemoryBarrier2<'static>, &'graph Arena>,
}

/// Synchronization on the main queue.
#[derive(Debug)]
pub struct QueueSync<'graph> {
	/// Pipeline barriers to execute.
	pub barriers: DependencyInfo<'graph>,
}

/// Synchronization between passes.
#[derive(Debug)]
pub struct Sync<'graph> {
	/// Synchronization on the main queue. This is unordered relative to the cross-queue synchronization.
	pub queue: QueueSync<'graph>,
	/// Any cross-queue synchronization that may be required.
	pub cross_queue: CrossQueueSync<'graph>,
}

/// State of CPU-side data output by a pass.
#[derive(Copy, Clone)]
pub enum DataState {
	/// Uninit, or has been moved out of by `PassContext::get_data`.
	Uninit,
	/// Init, possible referenced by `PassContext::get_data_ref`.
	Init { drop: fn(NonNull<()>) },
}

/// Concrete render graph resources.
pub enum Resource<'graph> {
	Data(NonNull<()>, DataState),
	Buffer(BufferData<'graph>),
	Image(ImageData<'graph>),
}

impl<'graph> Resource<'graph> {
	pub unsafe fn data<T>(&mut self) -> (NonNull<T>, &mut DataState) {
		match self {
			Resource::Data(ptr, state) => (ptr.cast(), state),
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn buffer(&self) -> &BufferData<'graph> {
		match self {
			Resource::Buffer(res) => res,
			Resource::Image(_) => unreachable!("expected buffer got image"),
			Resource::Data(..) => unreachable!("expected buffer got cpu data"),
		}
	}

	pub unsafe fn buffer_mut(&mut self) -> &mut BufferData<'graph> {
		match self {
			Resource::Buffer(res) => res,
			Resource::Image(_) => unreachable!("expected buffer got image"),
			Resource::Data(..) => unreachable!("expected buffer got cpu data"),
		}
	}

	pub unsafe fn image(&self) -> &ImageData<'graph> {
		match self {
			Resource::Image(res) => res,
			Resource::Buffer(_) => unreachable!("expected image got buffer"),
			Resource::Data(..) => unreachable!("expected image got cpu data"),
		}
	}

	pub unsafe fn image_mut(&mut self) -> &mut ImageData<'graph> {
		match self {
			Resource::Image(res) => res,
			Resource::Buffer(_) => unreachable!("expected image got buffer"),
			Resource::Data(..) => unreachable!("expected image got cpu data"),
		}
	}

	pub fn uninit(&self) -> bool {
		match self {
			Resource::Data(_, DataState::Uninit) => true,
			Resource::Data(_, DataState::Init { .. }) => false,
			Resource::Buffer(b) => b.uninit,
			Resource::Image(i) => i.uninit,
		}
	}
}

/// A map from virtual resources to (aliased) concrete resources.
pub struct ResourceMap<'graph> {
	resource_map: Vec<u32, &'graph Arena>,
	resources: Vec<Resource<'graph>, &'graph Arena>,
	buffers: Vec<u32, &'graph Arena>,
	images: Vec<u32, &'graph Arena>,
}

impl<'graph> ResourceMap<'graph> {
	unsafe fn new(
		resource_map: Vec<u32, &'graph Arena>, resources: Vec<Resource<'graph>, &'graph Arena>,
		buffers: Vec<u32, &'graph Arena>, images: Vec<u32, &'graph Arena>,
	) -> Self {
		Self {
			resource_map,
			resources,
			buffers,
			images,
		}
	}

	fn map_res(&self, res: u32) -> u32 {
		*self
			.resource_map
			.get(res as usize)
			.expect("resource ID from previous frame used")
	}

	fn arena(&self) -> &'graph Arena { self.resources.allocator() }

	fn buffers(&self) -> impl Iterator<Item = &BufferData<'graph>> {
		self.buffers.iter().map(move |&id| unsafe {
			let res = self.resources.get_unchecked(id as usize);
			res.buffer()
		})
	}

	fn images(&self) -> impl Iterator<Item = &ImageData<'graph>> {
		self.images.iter().map(move |&id| unsafe {
			let res = self.resources.get_unchecked(id as usize);
			res.image()
		})
	}

	pub fn cleanup(self) -> usize {
		for resource in self.resources {
			match resource {
				Resource::Data(ptr, state) => {
					if let DataState::Init { drop } = state {
						drop(ptr);
					}
					unsafe { Arena::deallocate(self.resource_map.allocator(), ptr.cast()) }
				},
				// Handled by cache reset.
				Resource::Image { .. } | Resource::Buffer { .. } => {},
			}
		}
		self.resource_map.len()
	}

	pub fn get(&mut self, res: u32) -> &'_ mut Resource<'graph> {
		let i = self.map_res(res) as usize;
		unsafe { self.resources.get_unchecked_mut(i) }
	}
}

fn usage_flags<T, U>(iter: impl IntoIterator<Item = T>) -> U
where
	T: Into<U>,
	U: BitOr<Output = U> + Default,
{
	iter.into_iter()
		.map(|x| x.into())
		.fold(Default::default(), |a, b| a | b)
}

struct ResourceAliaser<'graph> {
	buffers: Vec<u32, &'graph Arena>,
	images: ArenaMap<'graph, ImageDesc, Vec<u32, &'graph Arena>>,
	resource_map: Vec<u32, &'graph Arena>,
	resources: Vec<Resource<'graph>, &'graph Arena>,
	lifetimes: Vec<ResourceLifetime, &'graph Arena>,
}

impl<'graph> ResourceAliaser<'graph> {
	fn new(arena: &'graph Arena) -> Self {
		Self {
			buffers: Vec::new_in(arena),
			images: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
			resource_map: Vec::new_in(arena),
			resources: Vec::new_in(arena),
			lifetimes: Vec::new_in(arena),
		}
	}

	fn push(&mut self, desc: Resource<'graph>, lifetime: ResourceLifetime) {
		self.resource_map.push(self.resources.len() as u32);
		self.resources.push(desc);
		self.lifetimes.push(lifetime);
	}

	fn is_buffer_merge_candidate(data: &BufferData) -> bool {
		data.handle.buffer == vk::Buffer::null()
			&& matches!(data.desc.loc, BufferLoc::GpuOnly)
			&& data.desc.persist.is_none()
	}

	fn try_merge_buffer(&mut self, data: BufferData<'graph>, lifetime: ResourceLifetime) {
		// If the data to be merged is an external resource, don't try to merge it at all.
		if Self::is_buffer_merge_candidate(&data) {
			for &i in self.buffers.iter() {
				let res = &mut self.resources[i as usize];
				let res = unsafe { res.buffer_mut() };
				let res_lifetime = &mut self.lifetimes[i as usize];
				// If the lifetimes aren't overlapping, merge.
				if res_lifetime.independent(lifetime) && Self::is_buffer_merge_candidate(&res) {
					res.desc.size = res.desc.size.max(data.desc.size);
					res.usages.extend(data.usages);
					*res_lifetime = res_lifetime.union(lifetime);
					self.resource_map.push(i);
					return;
				}
			}
		}
		self.buffers.push(self.resources.len() as _);
		self.push(Resource::Buffer(data), lifetime);
	}

	fn is_image_merge_candidate(data: &ImageData) -> bool {
		data.handle.0 == vk::Image::null() && data.desc.persist.is_none()
	}

	fn try_merge_image(&mut self, data: ImageData<'graph>, lifetime: ResourceLifetime) {
		if Self::is_image_merge_candidate(&data) {
			for &i in self.images.get(&data.desc).into_iter().flatten() {
				let res = &mut self.resources[i as usize];
				let res = unsafe { res.image_mut() };
				let res_lifetime = &mut self.lifetimes[i as usize];
				// If the formats aren't compatible, don't merge.
				if res_lifetime.independent(lifetime)
					&& compatible_formats(
						res.usages.first_key_value().unwrap().1.format,
						data.usages.first_key_value().unwrap().1.format,
					) && Self::is_image_merge_candidate(&res)
				{
					res.usages.extend(data.usages);
					*res_lifetime = res_lifetime.union(lifetime);
					self.resource_map.push(i);
					return;
				}
			}
		}
		self.images
			.entry(data.desc)
			.or_insert_with(|| Vec::new_in(self.resources.allocator()))
			.push(self.resources.len() as _);
		self.push(Resource::Image(data), lifetime);
	}

	fn add(&mut self, resource: VirtualResourceData<'graph>) {
		match resource.ty {
			VirtualResourceType::Data(p) => self.push(Resource::Data(p, DataState::Uninit), resource.lifetime),
			VirtualResourceType::Buffer(data) => self.try_merge_buffer(data, resource.lifetime),
			VirtualResourceType::Image(data) => self.try_merge_image(data, resource.lifetime),
		}
	}

	fn finish(mut self, device: &Device, graph: &mut RenderGraph) -> ResourceMap<'graph> {
		let alloc = *self.resources.allocator();
		let mut buffers = Vec::new_in(alloc);
		let mut images = Vec::new_in(alloc);

		for (i, res) in self.resources.iter_mut().enumerate() {
			match res {
				Resource::Data(..) => {},
				Resource::Buffer(data) => {
					buffers.push(i as _);
					if data.handle.buffer == vk::Buffer::null() {
						let desc = crate::resource::BufferDescUnnamed {
							size: data.desc.size,
							usage: usage_flags(data.usages.values().flat_map(|x| x.usages.iter().copied())),
							readback: matches!(data.desc.loc, BufferLoc::Readback),
						};
						let per_desc = |name| crate::resource::BufferDesc {
							name,
							size: desc.size,
							usage: desc.usage,
							readback: desc.readback,
						};
						(data.handle, data.uninit) = match (data.desc.loc, data.desc.persist) {
							(BufferLoc::GpuOnly, Some(name)) => {
								let x = graph
									.caches
									.persistent_buffers
									.get(device, per_desc(name), vk::ImageLayout::UNDEFINED)
									.expect("failed to allocated graph buffer");
								(x.0, x.1)
							},
							(BufferLoc::GpuOnly, None) => graph
								.caches
								.buffers
								.get(device, desc)
								.expect("failed to allocated graph buffer"),
							(BufferLoc::Upload, x) => {
								assert!(x.is_none(), "cannot persist upload buffers");
								graph.caches.upload_buffers[graph.curr_frame]
									.get(device, desc)
									.expect("failed to allocated graph buffer")
							},
							(BufferLoc::Readback, x) => {
								let name = x.expect("readback buffers must be persistent");
								let x = graph.caches.readback_buffers[graph.curr_frame]
									.get(device, per_desc(name), vk::ImageLayout::UNDEFINED)
									.expect("failed to allocated graph buffer");
								(x.0, x.1)
							},
						};
					}
				},
				Resource::Image(data) => {
					images.push(i as _);
					if data.handle.0 == vk::Image::null() {
						let flags = data
							.usages
							.values()
							.any(|u| u.format != data.desc.format)
							.then_some(vk::ImageCreateFlags::MUTABLE_FORMAT)
							.unwrap_or_default();
						let desc = crate::resource::ImageDescUnnamed {
							flags,
							format: data.desc.format,
							size: data.desc.size,
							levels: data.desc.levels,
							layers: data.desc.layers,
							samples: data.desc.samples,
							usage: usage_flags(data.usages.values().flat_map(|x| x.usages.iter().copied())),
						};
						(data.handle, data.uninit) = if let Some(name) = data.desc.persist {
							let next_layout = data.usages.last_key_value().unwrap().1.as_prev().image_layout;
							let x = graph
								.caches
								.persistent_images
								.get(
									device,
									crate::resource::ImageDesc {
										name,
										flags,
										format: desc.format,
										size: desc.size,
										levels: desc.levels,
										layers: desc.layers,
										samples: desc.samples,
										usage: desc.usage,
									},
									next_layout,
								)
								.expect("failed to allocate graph image");
							((x.0, x.2), x.1)
						} else {
							let x = graph
								.caches
								.images
								.get(device, desc)
								.expect("failed to allocate graph image");
							((x.0, vk::ImageLayout::UNDEFINED), x.1)
						};
					}
				},
			}
		}

		unsafe { ResourceMap::new(self.resource_map, self.resources, buffers, images) }
	}
}

#[derive(Clone, Eq, PartialEq, Hash, Default)]
struct SyncPair<T> {
	from: T,
	to: T,
}

#[derive(Clone, PartialEq)]
struct InProgressImageBarrier {
	sync: SyncPair<AccessInfo>,
	subresource: Subresource,
}

#[derive(Clone, PartialEq)]
struct InProgressDependencyInfo<'graph> {
	barriers: ArenaMap<'graph, SyncPair<vk::PipelineStageFlags2>, SyncPair<vk::AccessFlags2>>,
	image_barriers: ArenaMap<'graph, vk::Image, InProgressImageBarrier>,
}

#[derive(Clone, PartialEq)]
struct InProgressCrossQueueSync<'graph> {
	signal: Vec<SyncStage<vk::Semaphore>, &'graph Arena>,
	signal_barriers: InProgressDependencyInfo<'graph>,
	wait: QueueWaitOwned<&'graph Arena>,
	wait_barriers: InProgressDependencyInfo<'graph>,
}

#[derive(Clone, PartialEq)]
struct InProgressSync<'graph> {
	queue: InProgressDependencyInfo<'graph>,
	cross_queue: InProgressCrossQueueSync<'graph>,
}

impl<'graph> InProgressDependencyInfo<'graph> {
	fn finish(self, signal: bool, wait: bool) -> DependencyInfo<'graph> {
		let arena = *self.barriers.allocator();
		DependencyInfo {
			barriers: self
				.barriers
				.into_iter()
				.map(|(s, a)| {
					GlobalBarrierAccess {
						previous_access: if !wait {
							AccessInfo {
								stage_mask: s.from,
								access_mask: a.from,
								image_layout: vk::ImageLayout::UNDEFINED,
							}
						} else {
							AccessInfo::default()
						},
						next_access: if !signal {
							AccessInfo {
								stage_mask: s.to,
								access_mask: a.to,
								image_layout: vk::ImageLayout::UNDEFINED,
							}
						} else {
							AccessInfo::default()
						},
					}
					.into()
				})
				.collect_in(arena),
			image_barriers: self
				.image_barriers
				.into_iter()
				.map(|(i, b)| {
					ImageBarrierAccess {
						previous_access: if !wait {
							b.sync.from
						} else {
							AccessInfo {
								image_layout: b.sync.from.image_layout,
								..Default::default()
							}
						},
						next_access: if !signal {
							b.sync.to
						} else {
							AccessInfo {
								image_layout: b.sync.to.image_layout,
								..Default::default()
							}
						},
						image: i,
						range: vk::ImageSubresourceRange {
							aspect_mask: b.subresource.aspect,
							base_array_layer: b.subresource.first_layer,
							layer_count: b.subresource.layer_count,
							base_mip_level: b.subresource.first_mip,
							level_count: b.subresource.mip_count,
						},
					}
					.into()
				})
				.collect_in(arena),
		}
	}
}

impl<'graph> InProgressCrossQueueSync<'graph> {
	fn finish(self) -> CrossQueueSync<'graph> {
		CrossQueueSync {
			signal: self.signal,
			signal_barriers: self.signal_barriers.finish(true, false),
			wait: self.wait,
			wait_barriers: self.wait_barriers.finish(false, true),
		}
	}
}

impl<'graph> InProgressSync<'graph> {
	fn finish(self) -> Sync<'graph> {
		Sync {
			queue: QueueSync {
				barriers: self.queue.finish(false, false),
			},
			cross_queue: self.cross_queue.finish(),
		}
	}
}

impl<'graph> InProgressDependencyInfo<'graph> {
	fn default(arena: &'graph Arena) -> Self {
		Self {
			barriers: ArenaMap::with_hasher_in(Default::default(), arena),
			image_barriers: ArenaMap::with_hasher_in(Default::default(), arena),
		}
	}
}

struct SyncBuilder<'temp, 'pass, 'graph> {
	sync: Vec<InProgressSync<'graph>, &'graph Arena>,
	passes: &'temp Vec<FrameEvent<'pass, 'graph>, &'graph Arena>,
}

impl<'temp, 'pass, 'graph> SyncBuilder<'temp, 'pass, 'graph> {
	fn new(arena: &'graph Arena, passes: &'temp Vec<FrameEvent<'pass, 'graph>, &'graph Arena>) -> Self {
		Self {
			sync: std::iter::repeat(InProgressSync {
				queue: InProgressDependencyInfo::default(arena),
				cross_queue: InProgressCrossQueueSync {
					signal: Vec::new_in(arena),
					signal_barriers: InProgressDependencyInfo::default(arena),
					wait: QueueWaitOwned::default(arena),
					wait_barriers: InProgressDependencyInfo::default(arena),
				},
			})
			.take(passes.len() + 1)
			.collect_in(arena),
			passes,
		}
	}

	/// Create a barrier between two passes.
	///
	/// If a global barrier is required, pass `Image::null()` and `ImageAspectFlags::empty()`.
	/// If no layout transition is required, an image barrier will be converted to a global barrier.
	fn barrier(
		&mut self, image: vk::Image, subresource: Subresource, _prev_pass: u32, prev_access: AccessInfo,
		next_pass: u32, next_access: AccessInfo,
	) {
		// As late as possible.
		let next = Self::before_pass(next_pass);
		let dep_info = &mut self.sync[next as usize].queue;
		Self::insert_info(dep_info, image, subresource, prev_access, next_access);
	}

	fn init_layout(&mut self, image: vk::Image, layout: vk::ImageLayout, subresource: Subresource, access: AccessInfo) {
		if image != vk::Image::null()
			&& access.image_layout != vk::ImageLayout::UNDEFINED
			&& access.image_layout != layout
		{
			let dep_info = &mut self.sync[Self::before_pass(0) as usize].queue;
			Self::insert_info(
				dep_info,
				image,
				subresource,
				AccessInfo {
					image_layout: layout,
					..Default::default()
				},
				access,
			);
		}
	}

	/// Handle the sync a swapchain requires.
	fn swapchain(
		&mut self, image: vk::Image, pass: u32, as_prev: AccessInfo, as_next: AccessInfo, available: vk::Semaphore,
		rendered: vk::Semaphore,
	) {
		let sync = Self::before_pass(pass);
		let cross_queue = &mut self.sync[sync as usize].cross_queue;
		// Layout transition to pass.
		Self::insert_info(
			&mut cross_queue.wait_barriers,
			image,
			Subresource::default(),
			AccessInfo::default(),
			as_next,
		);
		// Fire off the pass once the image is available.
		cross_queue.wait.binary_semaphores.push(SyncStage {
			point: available,
			stage: as_next.stage_mask,
		});

		let sync = Self::after_pass(pass);
		let cross_queue = &mut self.sync[sync as usize].cross_queue;
		// Layout transition to present.
		Self::insert_info(
			&mut cross_queue.signal_barriers,
			image,
			Subresource::default(),
			as_prev,
			get_access_info(UsageType::Present),
		);
		// Fire off present once finished.
		cross_queue.signal.push(SyncStage {
			point: rendered,
			stage: as_prev.stage_mask,
		})
	}

	#[inline]
	fn insert_info(
		dep_info: &mut InProgressDependencyInfo<'graph>, image: vk::Image, subresource: Subresource,
		prev_access: AccessInfo, mut next_access: AccessInfo,
	) {
		if next_access.stage_mask == vk::PipelineStageFlags2::empty() {
			// Ensure that the stage masks are valid if no stages were determined
			next_access.stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
		}

		if prev_access.image_layout == next_access.image_layout
			|| next_access.image_layout == vk::ImageLayout::UNDEFINED
			|| image == vk::Image::null()
		{
			// No transition required, use a global barrier instead.
			let access = dep_info
				.barriers
				.entry(SyncPair {
					from: prev_access.stage_mask,
					to: next_access.stage_mask,
				})
				.or_default();
			access.from |= prev_access.access_mask;
			access.to |= next_access.access_mask;
		} else {
			let old = dep_info.image_barriers.insert(
				image,
				InProgressImageBarrier {
					sync: SyncPair {
						from: prev_access,
						to: next_access,
					},
					subresource,
				},
			);
			debug_assert!(old.is_none(), "Multiple transitions for one image in a single pass");
		}
	}

	fn before_pass(pass: u32) -> u32 { pass }

	fn after_pass(pass: u32) -> u32 { pass + 1 }

	fn finish(self) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let arena = *self.sync.allocator();
		let mut all_sync: Vec<_, _> = self.sync.into_iter().map(|x| x.finish()).collect_in(arena);

		if let Some(sync) = all_sync.last_mut() {
			sync.queue.barriers.barriers.push(
				GlobalBarrier {
					previous_usages: &[UsageType::General],
					next_usages: &[UsageType::HostRead],
				}
				.into(),
			);
		}

		Ok(all_sync)
	}
}

trait Usage {
	fn inner(&self) -> impl Iterator<Item = UsageType>;

	fn subresource(&self) -> Subresource;

	fn as_prev(&self) -> AccessInfo { as_previous_access(self.inner(), false) }

	fn as_next(&self, prev_access: AccessInfo) -> AccessInfo { as_next_access(self.inner(), prev_access) }

	fn is_write(&self) -> bool { self.inner().any(is_write_access) }
}
impl<A: Allocator> Usage for BufferUsageOwned<A> {
	fn inner(&self) -> impl Iterator<Item = UsageType> { self.usages.iter().map(|&x| x.into()) }

	fn subresource(&self) -> Subresource { Subresource::default() }
}
impl<A: Allocator> Usage for ImageUsageOwned<A> {
	fn inner(&self) -> impl Iterator<Item = UsageType> { self.usages.iter().map(|&x| x.into()) }

	fn subresource(&self) -> Subresource { self.subresource }
}

trait ToImage: Copy {
	fn to_image(self) -> (vk::Image, vk::ImageLayout);
}
impl ToImage for BufferHandle {
	fn to_image(self) -> (vk::Image, vk::ImageLayout) { (vk::Image::null(), vk::ImageLayout::UNDEFINED) }
}
impl ToImage for (vk::Image, vk::ImageLayout) {
	fn to_image(self) -> (vk::Image, vk::ImageLayout) { self }
}

struct Synchronizer<'temp, 'pass, 'graph> {
	resource_map: &'temp ResourceMap<'graph>,
	passes: &'temp Vec<FrameEvent<'pass, 'graph>, &'graph Arena>,
}

impl<'temp, 'pass, 'graph> Synchronizer<'temp, 'pass, 'graph> {
	fn new(
		resource_map: &'temp ResourceMap<'graph>, passes: &'temp Vec<FrameEvent<'pass, 'graph>, &'graph Arena>,
	) -> Self {
		Self { resource_map, passes }
	}

	fn do_sync_for<D, H: ToImage, U: Usage>(&mut self, sync: &mut SyncBuilder, res: &GpuData<D, H, U>) {
		let mut usages = res.usages.iter().peekable();
		if let Some((available, rendered)) = res.swapchain {
			let (&pass, usage) = usages.next().unwrap();
			assert!(usages.next().is_none(), "Swapchains can only be used in one pass");
			sync.swapchain(
				res.handle.to_image().0,
				pass,
				usage.as_prev(),
				usage.as_next(AccessInfo::default()),
				available,
				rendered,
			);
			return;
		}

		let (&(mut prev_pass), usage) = usages.next().unwrap();
		let mut prev_access = usage.as_prev();
		let (i, l) = res.handle.to_image();
		sync.init_layout(i, l, usage.subresource(), usage.as_next(AccessInfo::default()));

		while let Some((&pass, usage)) = usages.next() {
			let next_pass = pass;
			let mut next_access = usage.as_next(prev_access);
			let mut next_prev_access = usage.as_prev();

			if usage.is_write() {
				// We're a write, so we can't merge future passes into us.
				sync.barrier(
					res.handle.to_image().0,
					usage.subresource(),
					prev_pass,
					prev_access,
					next_pass,
					next_access,
				);
				prev_pass = next_pass;
				prev_access = next_prev_access;
			} else {
				// We're a read, let's look ahead and merge any other reads into us.
				let mut last_read_pass = next_pass;
				let mut subresource = usage.subresource();
				while let Some((&pass, usage)) = usages.peek() {
					let as_next = usage.as_next(prev_access);
					if usage.is_write() || as_next.image_layout != prev_access.image_layout {
						// We've hit a write, stop merging now.
						// Note that read -> read layout transitions are also writes.
						break;
					}
					last_read_pass = pass;
					next_access |= as_next;
					next_prev_access |= usage.as_prev();
					let sub = usage.subresource();
					subresource.aspect |= sub.aspect;

					let curr_last_layer = subresource.first_layer + subresource.layer_count;
					let next_last_layer = sub.first_layer + sub.layer_count;
					let last_layer = curr_last_layer.max(next_last_layer);
					subresource.first_layer = subresource.first_layer.min(sub.first_layer);
					subresource.layer_count = last_layer - subresource.first_layer;

					let curr_last_mip = subresource.first_mip + subresource.mip_count;
					let next_last_mip = sub.first_mip + sub.mip_count;
					let last_mip = curr_last_mip.max(next_last_mip);
					subresource.first_mip = subresource.first_mip.min(sub.first_mip);
					subresource.mip_count = last_mip - subresource.first_mip;

					usages.next();
				}

				sync.barrier(
					res.handle.to_image().0,
					subresource,
					prev_pass,
					prev_access,
					next_pass,
					next_access,
				);
				prev_pass = last_read_pass;
				prev_access = next_prev_access;
			}
		}
	}

	fn sync(&mut self) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let mut sync = SyncBuilder::new(self.resource_map.arena(), self.passes);

		for buffer in self.resource_map.buffers() {
			self.do_sync_for(&mut sync, buffer);
		}

		for image in self.resource_map.images() {
			self.do_sync_for(&mut sync, image)
		}

		sync.finish()
	}
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub(super) fn compile(self, device: &Device, arena: &'graph Arena) -> Result<CompiledFrame<'pass, 'graph>> {
		let span = span!(Level::TRACE, "compile graph");
		let _e = span.enter();

		// The order of the passes is already topologically sorted.
		// This is order we will run them in.

		let resource_map = {
			let span = span!(Level::TRACE, "alias resources");
			let _e = span.enter();

			self.virtual_resources
				.into_iter()
				.fold(ResourceAliaser::new(arena), |mut a, r| {
					a.add(r);
					a
				})
				.finish(device, self.graph)
		};

		let sync = {
			let span = span!(Level::TRACE, "synchronize");
			let _e = span.enter();

			Synchronizer::new(&resource_map, &self.passes).sync()
		}?;

		Ok(CompiledFrame {
			passes: self.passes,
			sync,
			resource_map,
			graph: self.graph,
		})
	}
}
