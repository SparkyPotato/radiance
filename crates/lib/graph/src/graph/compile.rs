use std::{alloc::Allocator, hash::BuildHasherDefault, hint::unreachable_unchecked, ops::BitOr, ptr::NonNull};

use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	device::{Device, QueueWaitOwned, SyncStage},
	graph::{
		cache::ResourceList,
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
		Frame,
		ImageDesc,
		PassData,
		RenderGraph,
	},
	resource::{BufferHandle, Subresource},
	sync::{
		as_next_access,
		as_previous_access,
		get_access_info,
		is_write_access,
		AccessInfo,
		GlobalBarrierAccess,
		ImageBarrierAccess,
		UsageType,
	},
	Result,
};

pub(super) struct CompiledFrame<'pass, 'graph> {
	pub passes: Vec<PassData<'pass, 'graph>, &'graph Arena>,
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

#[derive(Debug)]
pub struct EventInfo<'graph> {
	pub event: vk::Event,
	pub info: DependencyInfo<'graph>,
}

/// Synchronization on the main queue.
#[derive(Debug)]
pub struct QueueSync<'graph> {
	/// Pipeline barriers to execute.
	pub barriers: DependencyInfo<'graph>,
	/// Events to wait on.
	pub wait_events: Vec<EventInfo<'graph>, &'graph Arena>,
	/// Events to set.
	pub set_events: Vec<EventInfo<'graph>, &'graph Arena>,
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

	fn try_merge_buffer(&mut self, data: BufferData<'graph>, lifetime: ResourceLifetime) {
		// If the data to be merged is an external resource, don't try to merge it at all.
		if data.handle.buffer != vk::Buffer::null() {
			for &i in self.buffers.iter() {
				let res = &mut self.resources[i as usize];
				let res_lifetime = &mut self.lifetimes[i as usize];
				let buffer = unsafe { res.buffer_mut() };
				// If the lifetimes aren't overlapping, merge.
				if !res_lifetime.independent(lifetime) {
					continue;
				}

				buffer.desc.size = buffer.desc.size.max(data.desc.size);
				buffer.usages.extend(data.usages);
				*res_lifetime = res_lifetime.union(lifetime);
				self.resource_map.push(i);

				return;
			}
		}
		self.buffers.push(self.resources.len() as _);
		self.push(Resource::Buffer(data), lifetime);
	}

	fn try_merge_image(&mut self, data: ImageData<'graph>, lifetime: ResourceLifetime) {
		if data.handle != vk::Image::null() {
			for &i in self.images.get(&data.desc).into_iter().flatten() {
				let res = &mut self.resources[i as usize];
				let res_lifetime = &mut self.lifetimes[i as usize];
				let image = unsafe { res.image_mut() };
				// If the formats aren't compatible, don't merge.
				if !res_lifetime.independent(lifetime)
					|| compatible_formats(
						image.usages.first_key_value().unwrap().1.format,
						data.usages.first_key_value().unwrap().1.format,
					) {
					continue;
				}

				image.usages.extend(data.usages);
				*res_lifetime = res_lifetime.union(lifetime);
				self.resource_map.push(i);

				return;
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
						data.handle = if data.desc.upload {
							&mut graph.caches.upload_buffers[graph.curr_frame]
						} else {
							&mut graph.caches.buffers
						}
						.get(
							device,
							crate::resource::BufferDescUnnamed {
								size: data.desc.size,
								usage: usage_flags(data.usages.values().flat_map(|x| x.usages.iter().copied())),
								on_cpu: false,
							},
						)
						.expect("failed to allocated graph buffer");
					}
				},

				Resource::Image(data) => {
					images.push(i as _);
					if data.handle == vk::Image::null() {
						let flags = data
							.usages
							.values()
							.any(|u| u.format != data.desc.format)
							.then_some(vk::ImageCreateFlags::MUTABLE_FORMAT)
							.unwrap_or_default();
						data.handle = graph
							.caches
							.images
							.get(
								device,
								crate::resource::ImageDescUnnamed {
									flags,
									format: data.desc.format,
									size: data.desc.size,
									levels: data.desc.levels,
									layers: data.desc.layers,
									samples: data.desc.samples,
									usage: usage_flags(data.usages.values().flat_map(|x| x.usages.iter().copied())),
								},
							)
							.expect("failed to allocate graph image");
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

#[derive(Clone)]
struct InProgressImageBarrier {
	sync: SyncPair<AccessInfo>,
	subresource: Subresource,
	qfot: Option<u32>,
}

#[derive(Clone)]
struct InProgressDependencyInfo<'graph> {
	barriers: ArenaMap<'graph, SyncPair<vk::PipelineStageFlags2>, SyncPair<vk::AccessFlags2>>,
	image_barriers: ArenaMap<'graph, vk::Image, InProgressImageBarrier>,
}

#[derive(Clone)]
struct InProgressCrossQueueSync<'graph> {
	signal: Vec<SyncStage<vk::Semaphore>, &'graph Arena>,
	signal_barriers: InProgressDependencyInfo<'graph>,
	wait: QueueWaitOwned<&'graph Arena>,
	wait_barriers: InProgressDependencyInfo<'graph>,
}

#[derive(Clone)]
struct InProgressSync<'graph> {
	queue: InProgressDependencyInfo<'graph>,
	cross_queue: InProgressCrossQueueSync<'graph>,
}

enum Qfot {
	From,
	To,
	None,
}

impl<'graph> InProgressDependencyInfo<'graph> {
	fn finish(self, device: &Device, qfot: Qfot) -> DependencyInfo<'graph> {
		let arena = *self.barriers.allocator();
		DependencyInfo {
			barriers: self
				.barriers
				.into_iter()
				.map(|(s, a)| {
					GlobalBarrierAccess {
						previous_access: AccessInfo {
							stage_mask: s.from,
							access_mask: a.from,
							image_layout: vk::ImageLayout::UNDEFINED,
						},
						next_access: AccessInfo {
							stage_mask: s.to,
							access_mask: a.to,
							image_layout: vk::ImageLayout::UNDEFINED,
						},
					}
					.into()
				})
				.collect_in(arena),
			image_barriers: self
				.image_barriers
				.into_iter()
				.map(|(i, b)| {
					let (src_queue_family_index, previous_access, dst_queue_family_index, next_access) = match qfot {
						Qfot::From if b.qfot.is_some() => (
							b.qfot.unwrap(),
							AccessInfo::default(),
							device.queue_families().graphics,
							b.sync.to,
						),
						Qfot::To if b.qfot.is_some() => (
							device.queue_families().graphics,
							b.sync.from,
							b.qfot.unwrap(),
							AccessInfo::default(),
						),
						Qfot::None => {
							debug_assert!(b.qfot.is_none(), "QFOT not allowed in main queue sync");
							(0, b.sync.from, 0, b.sync.to)
						},
						_ => (0, b.sync.from, 0, b.sync.to),
					};
					ImageBarrierAccess {
						previous_access,
						next_access,
						image: i,
						range: vk::ImageSubresourceRange {
							aspect_mask: b.subresource.aspect,
							base_array_layer: b.subresource.first_layer,
							layer_count: b.subresource.layer_count,
							base_mip_level: b.subresource.first_mip,
							level_count: b.subresource.mip_count,
						},
						src_queue_family_index,
						dst_queue_family_index,
					}
					.into()
				})
				.collect_in(arena),
		}
	}
}

impl<'graph> InProgressCrossQueueSync<'graph> {
	fn finish(self, device: &Device) -> CrossQueueSync<'graph> {
		CrossQueueSync {
			signal: self.signal,
			signal_barriers: self.signal_barriers.finish(device, Qfot::To),
			wait: self.wait,
			wait_barriers: self.wait_barriers.finish(device, Qfot::From),
		}
	}
}

impl<'graph> InProgressSync<'graph> {
	fn finish(self, device: &Device) -> Sync<'graph> {
		let arena = *self.queue.barriers.allocator();
		Sync {
			queue: QueueSync {
				barriers: self.queue.finish(device, Qfot::None),
				set_events: Vec::new_in(arena),
				wait_events: Vec::new_in(arena),
			},
			cross_queue: self.cross_queue.finish(device),
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

struct SyncBuilder<'graph> {
	sync: Vec<InProgressSync<'graph>, &'graph Arena>,
	events: ArenaMap<'graph, SyncPair<u32>, InProgressDependencyInfo<'graph>>,
}

impl<'graph> SyncBuilder<'graph> {
	fn new<'temp, 'pass>(arena: &'graph Arena, passes: &'temp [PassData<'pass, 'graph>]) -> Self {
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
			events: ArenaMap::with_hasher_in(Default::default(), arena),
		}
	}

	/// Create a barrier between two passes.
	///
	/// If a global barrier is required, pass `Image::null()` and `ImageAspectFlags::empty()`. If no layout transition
	/// is required, an image barrier will be converted to a global barrier.
	///
	/// This will internally use either a pipeline barrier or an event, depending on what is optimal.
	fn barrier(
		&mut self, image: vk::Image, subresource: Subresource, prev_pass: u32, prev_access: AccessInfo, next_pass: u32,
		next_access: AccessInfo,
	) {
		let dep_info = self.get_dep_info(prev_pass, next_pass);
		Self::insert_info(dep_info, image, subresource, prev_access, next_access, None);
	}

	fn init_layout(&mut self, image: vk::Image, subresource: Subresource, access: AccessInfo) {
		if access.image_layout != vk::ImageLayout::UNDEFINED {
			let dep_info = &mut self.sync[Self::before_pass(0) as usize].queue;
			Self::insert_info(dep_info, image, subresource, AccessInfo::default(), access, None);
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
			None,
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
			None,
		);
		// Fire off present once finished.
		cross_queue.signal.push(SyncStage {
			point: rendered,
			stage: as_prev.stage_mask,
		})
	}

	fn get_dep_info(&mut self, prev_pass: u32, next_pass: u32) -> &mut InProgressDependencyInfo<'graph> {
		let prev = Self::after_pass(prev_pass);
		let next = Self::before_pass(next_pass);

		if prev == next {
			// We can use a barrier here, since the resource will be used right after.
			&mut self.sync[prev as usize].queue
		} else {
			// Use an event, since there is some gap between the previous and next access.
			let arena = *self.events.allocator();
			self.events
				.entry(SyncPair { from: prev, to: next })
				.or_insert_with(|| InProgressDependencyInfo::default(arena))
		}
	}

	#[inline]
	fn insert_info(
		dep_info: &mut InProgressDependencyInfo<'graph>, image: vk::Image, subresource: Subresource,
		prev_access: AccessInfo, mut next_access: AccessInfo, qfot: Option<u32>,
	) {
		if next_access.stage_mask == vk::PipelineStageFlags2::empty() {
			// Ensure that the stage masks are valid if no stages were determined
			next_access.stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
		}

		if prev_access.image_layout == next_access.image_layout && qfot.is_none() || image == vk::Image::null() {
			// No transition or QFOT required, use a global barrier instead.
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
					qfot,
				},
			);
			debug_assert!(old.is_none(), "Multiple transitions for one image in a single pass");
		}
	}

	fn before_pass(pass: u32) -> u32 { pass }

	fn after_pass(pass: u32) -> u32 { pass + 1 }

	fn finish(
		self, device: &Device, event_list: &mut ResourceList<crate::resource::Event>,
	) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let arena = *self.sync.allocator();
		let mut sync: Vec<Sync, _> = self.sync.into_iter().map(|x| x.finish(device)).collect_in(arena);

		for (pair, info) in self.events {
			let event = event_list.get_or_create(device, ())?;
			let info = info.finish(device, Qfot::None);

			sync[pair.from as usize].queue.set_events.push(EventInfo {
				event,
				info: info.clone(),
			});
			sync[pair.to as usize].queue.wait_events.push(EventInfo { event, info });
		}

		Ok(sync)
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
	fn to_image(self) -> vk::Image;
}
impl ToImage for BufferHandle {
	fn to_image(self) -> vk::Image { vk::Image::null() }
}
impl ToImage for vk::Image {
	fn to_image(self) -> vk::Image { self }
}

struct Synchronizer<'temp, 'pass, 'graph> {
	resource_map: &'temp ResourceMap<'graph>,
	passes: &'temp [PassData<'pass, 'graph>],
}

impl<'temp, 'pass, 'graph> Synchronizer<'temp, 'pass, 'graph> {
	fn new(resource_map: &'temp ResourceMap<'graph>, passes: &'temp [PassData<'pass, 'graph>]) -> Self {
		Self { resource_map, passes }
	}

	fn do_sync_for<D, H: ToImage, U: Usage>(&mut self, sync: &mut SyncBuilder, res: &GpuData<D, H, U>) {
		let mut usages = res.usages.iter().peekable();
		if let Some((available, rendered)) = res.swapchain {
			let (&pass, usage) = usages.next().unwrap();
			assert!(usages.next().is_none(), "Swapchains can only be used in one pass");
			sync.swapchain(
				res.handle.to_image(),
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
		sync.init_layout(
			res.handle.to_image(),
			usage.subresource(),
			usage.as_next(AccessInfo::default()),
		);

		while let Some((&pass, usage)) = usages.next() {
			let next_pass = pass;
			let mut next_access = usage.as_next(prev_access);
			let mut next_prev_access = usage.as_prev();

			if usage.is_write() {
				// We're a write, so we can't merge future passes into us.
				sync.barrier(
					res.handle.to_image(),
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
					res.handle.to_image(),
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

	fn sync(
		&mut self, device: &Device, event_list: &mut ResourceList<crate::resource::Event>,
	) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let mut sync = SyncBuilder::new(self.resource_map.arena(), self.passes);

		for buffer in self.resource_map.buffers() {
			self.do_sync_for(&mut sync, buffer);
		}

		for image in self.resource_map.images() {
			self.do_sync_for(&mut sync, image)
		}

		sync.finish(device, event_list)
	}
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub(super) fn compile(self, device: &Device) -> Result<CompiledFrame<'pass, 'graph>> {
		let span = span!(Level::TRACE, "compile graph");
		let _e = span.enter();

		// The order of the passes is already topologically sorted.
		// This is order we will run them in.

		let resource_map = {
			let span = span!(Level::TRACE, "alias resources");
			let _e = span.enter();

			self.virtual_resources
				.into_iter()
				.fold(ResourceAliaser::new(self.device.arena()), |mut a, r| {
					a.add(r);
					a
				})
				.finish(device, self.graph)
		};

		let sync = {
			let span = span!(Level::TRACE, "synchronize");
			let _e = span.enter();

			Synchronizer::new(&resource_map, &self.passes).sync(device, &mut self.graph.caches.events)
		}?;

		Ok(CompiledFrame {
			passes: self.passes,
			sync,
			resource_map,
			graph: self.graph,
		})
	}
}
