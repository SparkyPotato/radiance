use std::{collections::BTreeMap, hash::BuildHasherDefault, hint::unreachable_unchecked, ops::BitOr, ptr::NonNull};

use ash::vk::{
	AccessFlags2,
	BufferUsageFlags,
	Event,
	Format,
	Image,
	ImageAspectFlags,
	ImageCreateFlags,
	ImageLayout,
	ImageMemoryBarrier2,
	ImageSubresourceRange,
	ImageUsageFlags,
	ImageViewType,
	MemoryBarrier2,
	PipelineStageFlags2,
	Semaphore,
	SemaphoreSubmitInfo,
	REMAINING_ARRAY_LAYERS,
	REMAINING_MIP_LEVELS,
};
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{
		cache::ResourceList,
		virtual_resource::{
			compatible_formats,
			BufferUsageOwned,
			GpuBufferType,
			GpuData,
			ImageType,
			ImageUsageOwned,
			ResourceLifetime,
			Usage,
			VirtualResourceData,
			VirtualResourceType,
		},
		ArenaMap,
		ExternalSync,
		Frame,
		GpuBufferHandle,
		PassData,
		RenderGraph,
		UploadBufferHandle,
	},
	resource,
	resource::{BufferDesc, ImageDesc},
	sync::{
		as_next_access,
		as_previous_access,
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

pub struct CrossQueueSync<'graph> {
	/// Semaphores to signal. Signals occur before waits.
	pub signal_semaphores: Vec<SemaphoreSubmitInfo, &'graph Arena>,
	/// The synchronization that happens before the signal.
	pub signal_barriers: DependencyInfo<'graph>,
	/// Semaphores to wait on. Waits occur after signals.
	pub wait_semaphores: Vec<SemaphoreSubmitInfo, &'graph Arena>,
	/// The synchronization that happens after the wait.
	pub wait_barriers: DependencyInfo<'graph>,
}

#[derive(Clone)]
pub struct DependencyInfo<'graph> {
	/// Global barriers.
	pub barriers: Vec<MemoryBarrier2, &'graph Arena>,
	/// Image barriers.
	pub image_barriers: Vec<ImageMemoryBarrier2, &'graph Arena>,
}

pub struct EventInfo<'graph> {
	pub event: Event,
	pub info: DependencyInfo<'graph>,
}

/// Synchronization on the main queue.
pub struct QueueSync<'graph> {
	/// Pipeline barriers to execute.
	pub barriers: DependencyInfo<'graph>,
	/// Events to wait on.
	pub wait_events: Vec<EventInfo<'graph>, &'graph Arena>,
	/// Events to set.
	pub set_events: Vec<EventInfo<'graph>, &'graph Arena>,
}

/// Synchronization between passes.
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

/// A GPU resource that keeps track of all the passes it is used in, in order.
pub struct GpuResource<'graph, H, U> {
	pub handle: H,
	pub usages: BTreeMap<u32, U, &'graph Arena>,
}

/// A GPU Resource that is also externally synchronized.
pub struct SyncedResource<'graph, H, U: Usage> {
	pub resource: GpuResource<'graph, H, U>,
	pub prev_usage: ExternalSync<Vec<U::Inner, &'graph Arena>>,
	pub next_usage: ExternalSync<Vec<U::Inner, &'graph Arena>>,
}

type BufferResource<'graph> = SyncedResource<'graph, GpuBufferHandle, BufferUsageOwned<'graph>>;
type ImageResource<'graph> = SyncedResource<'graph, Image, ImageUsageOwned<'graph>>;

/// Concrete render graph resources.
pub enum Resource<'graph> {
	Data(NonNull<()>, DataState),
	UploadBuffer(UploadBufferHandle),
	GpuBuffer(BufferResource<'graph>),
	Image(ImageResource<'graph>),
}

impl<'graph> Resource<'graph> {
	pub unsafe fn data<T>(&mut self) -> (NonNull<T>, &mut DataState) {
		match self {
			Resource::Data(ptr, state) => (ptr.cast(), state),
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn upload_buffer(&self) -> UploadBufferHandle {
		match self {
			Resource::UploadBuffer(h) => *h,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn gpu_buffer(&self) -> &BufferResource<'graph> {
		match self {
			Resource::GpuBuffer(res) => res,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn image(&self) -> &ImageResource<'graph> {
		match self {
			Resource::Image(res) => res,
			_ => unreachable_unchecked(),
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

	fn buffers(&self) -> impl Iterator<Item = &BufferResource<'graph>> {
		self.buffers.iter().map(move |&id| unsafe {
			let res = self.resources.get_unchecked(id as usize);
			res.gpu_buffer()
		})
	}

	fn images(&self) -> impl Iterator<Item = &ImageResource<'graph>> {
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
					unsafe { self.resource_map.allocator().deallocate(ptr.cast()) }
				},
				// Handled by cache reset.
				Resource::Image { .. } | Resource::GpuBuffer { .. } | Resource::UploadBuffer(_) => {},
			}
		}
		self.resource_map.len()
	}

	pub fn get(&mut self, res: u32) -> &'_ mut Resource<'graph> {
		let i = self.map_res(res) as usize;
		unsafe { self.resources.get_unchecked_mut(i) }
	}
}

/// A possible candidate for merging two virtual resources.
#[derive(Eq, PartialEq, Hash)]
enum MergeCandidate {
	GpuBuffer,
	Image(super::ImageDesc),
}

type GpuBufferResourceDesc<'graph> = GpuResource<'graph, BufferDesc, BufferUsageOwned<'graph>>;
type ImageResourceDesc<'graph> = GpuResource<'graph, ImageDesc, ImageUsageOwned<'graph>>;

/// A description for a concrete resource.
enum ResourceDescType<'graph> {
	Data(NonNull<()>),
	UploadBuffer(BufferDesc),
	GpuBuffer(GpuBufferResourceDesc<'graph>),
	Image(ImageResourceDesc<'graph>),
	ExternalBuffer(BufferResource<'graph>),
	ExternalImage(ImageResource<'graph>),
}

fn usage_flags<T, U>(iter: impl IntoIterator<Item = T>) -> U
where
	T: Into<U>,
	U: BitOr<Output = U> + Default,
{
	iter.into_iter()
		.map(|x| x.into())
		.reduce(|a: U, b| a | b)
		.unwrap_or_default()
}

impl<'graph> VirtualResourceType<'graph> {
	fn to_res(self, pass: u32) -> ResourceDescType<'graph> {
		match self {
			VirtualResourceType::Data(ptr) => ResourceDescType::Data(ptr),
			VirtualResourceType::UploadBuffer(desc) => ResourceDescType::UploadBuffer(BufferDesc {
				size: desc.desc,
				usage: usage_flags::<_, BufferUsageFlags>(
					desc.read_usages
						.iter()
						.map(|(_, x)| -> BufferUsageFlags { usage_flags(x.usages.iter().copied()) }),
				) | usage_flags(desc.write_usage.usages.iter().copied()),
			}),
			VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::Internal(size),
				write_usage,
				read_usages,
			}) => ResourceDescType::GpuBuffer(GpuResource {
				handle: BufferDesc {
					size,
					usage: usage_flags::<_, BufferUsageFlags>(
						read_usages
							.iter()
							.map(|(_, x)| -> BufferUsageFlags { usage_flags(x.usages.iter().copied()) }),
					) | usage_flags(write_usage.usages.iter().copied()),
				},
				usages: {
					let arena = *read_usages.allocator();
					read_usages
						.into_iter()
						.chain(std::iter::once((pass, write_usage)))
						.collect_in(arena)
				},
			}),
			VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::External(buf),
				write_usage,
				read_usages,
			}) => ResourceDescType::ExternalBuffer(BufferResource {
				resource: GpuResource {
					handle: buf.handle,
					usages: {
						let arena = *read_usages.allocator();
						read_usages
							.into_iter()
							.chain(std::iter::once((pass, write_usage)))
							.collect_in(arena)
					},
				},
				prev_usage: buf.prev_usage,
				next_usage: buf.next_usage,
			}),
			VirtualResourceType::Image(GpuData {
				desc: ImageType::Internal(desc),
				read_usages,
				write_usage,
			}) => {
				let mut usages = BTreeMap::new_in(*read_usages.allocator());

				let mut usage: ImageUsageFlags = usage_flags(write_usage.usages.iter().copied());
				let mut flags = write_usage.create_flags();
				let format = write_usage.format;
				usages.insert(pass, write_usage);

				for (pass, read) in read_usages {
					if read.format != format {
						flags |= ImageCreateFlags::MUTABLE_FORMAT;
					}

					let r: ImageUsageFlags = usage_flags(read.usages.iter().copied());
					usage |= r;
					flags |= read.create_flags();
					usages.insert(pass, read);
				}

				ResourceDescType::Image(GpuResource {
					handle: ImageDesc {
						flags,
						format,
						size: desc.size,
						levels: desc.levels,
						layers: desc.layers,
						samples: desc.samples,
						usage,
					},
					usages,
				})
			},
			VirtualResourceType::Image(GpuData {
				desc: ImageType::External(img),
				write_usage,
				read_usages,
			}) => {
				let mut usages = BTreeMap::new_in(*read_usages.allocator());

				usages.insert(pass, write_usage);

				for (pass, read) in read_usages {
					usages.insert(pass, read);
				}

				ResourceDescType::ExternalImage(ImageResource {
					resource: GpuResource {
						handle: img.handle,
						usages,
					},
					prev_usage: img.prev_usage,
					next_usage: img.next_usage,
				})
			},
		}
	}
}

impl<'graph> ResourceDescType<'graph> {
	unsafe fn gpu_buffer(&mut self) -> &mut GpuBufferResourceDesc<'graph> {
		match self {
			ResourceDescType::GpuBuffer(res) => res,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&mut self) -> &mut ImageResourceDesc<'graph> {
		match self {
			ResourceDescType::Image(res) => res,
			_ => unreachable_unchecked(),
		}
	}
}

struct ResourceDesc<'graph> {
	lifetime: ResourceLifetime,
	ty: ResourceDescType<'graph>,
}

impl<'graph> From<VirtualResourceData<'graph>> for ResourceDesc<'graph> {
	fn from(value: VirtualResourceData<'graph>) -> Self {
		Self {
			ty: value.ty.to_res(value.lifetime.start),
			lifetime: value.lifetime,
		}
	}
}

impl<'a> ResourceDesc<'a> {
	/// Returns `true` if the resource was merged.
	unsafe fn try_merge(&mut self, other: &mut VirtualResourceData<'a>) -> bool {
		if !self.lifetime.independent(other.lifetime) {
			return false;
		}

		let ret = match other.ty {
			VirtualResourceType::Data(_)
			| VirtualResourceType::UploadBuffer(_)
			| VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::External(_),
				..
			})
			| VirtualResourceType::Image(GpuData {
				desc: ImageType::External(_),
				..
			}) => unreachable_unchecked(),
			VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::Internal(size),
				ref mut write_usage,
				ref mut read_usages,
			}) => {
				let alloc = *write_usage.usages.allocator();

				let this = self.ty.gpu_buffer();
				this.handle.size = this.handle.size.max(size);
				let u: BufferUsageFlags = usage_flags(write_usage.usages.iter().copied());
				this.handle.usage |= u;
				this.usages.insert(
					other.lifetime.start,
					std::mem::replace(
						write_usage,
						BufferUsageOwned {
							usages: Vec::new_in(alloc),
						},
					),
				);

				for (pass, read) in std::mem::replace(read_usages, ArenaMap::with_hasher_in(Default::default(), alloc))
				{
					let r: BufferUsageFlags = usage_flags(write_usage.usages.iter().copied());
					this.handle.usage |= r;
					this.usages.insert(pass, read);
				}

				true
			},
			VirtualResourceType::Image(GpuData {
				desc: ImageType::Internal(_),
				ref mut write_usage,
				ref mut read_usages,
			}) => {
				let alloc = *write_usage.usages.allocator();

				let this = self.ty.image();
				if !compatible_formats(this.handle.format, write_usage.format) {
					return false;
				}

				let u: ImageUsageFlags = usage_flags(write_usage.usages.iter().copied());
				this.handle.usage |= u;
				this.handle.flags |= write_usage.create_flags();
				this.usages.insert(
					other.lifetime.start,
					std::mem::replace(
						write_usage,
						ImageUsageOwned {
							format: Format::UNDEFINED,
							usages: Vec::new_in(alloc),
							view_type: ImageViewType::TYPE_1D,
							aspect: ImageAspectFlags::empty(),
						},
					),
				);

				for (pass, read) in std::mem::replace(read_usages, ArenaMap::with_hasher_in(Default::default(), alloc))
				{
					let r: ImageUsageFlags = usage_flags(write_usage.usages.iter().copied());
					this.handle.usage |= r;
					this.handle.flags |= read.create_flags();
					this.usages.insert(pass, read);
				}

				true
			},
		};

		if ret {
			self.lifetime = self.lifetime.union(other.lifetime);
		}
		ret
	}
}

struct ResourceAliaser<'graph> {
	aliasable: ArenaMap<'graph, MergeCandidate, Vec<u32, &'graph Arena>>,
	resource_map: Vec<u32, &'graph Arena>,
	resources: Vec<ResourceDesc<'graph>, &'graph Arena>,
}

impl<'graph> ResourceAliaser<'graph> {
	fn new(arena: &'graph Arena) -> Self {
		Self {
			aliasable: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
			resources: Vec::new_in(arena),
			resource_map: Vec::new_in(arena),
		}
	}

	fn push(&mut self, desc: ResourceDesc<'graph>) {
		self.resource_map.push(self.resources.len() as u32);
		self.resources.push(desc);
	}

	unsafe fn merge(&mut self, merge: MergeCandidate, mut resource: VirtualResourceData<'graph>) {
		let merges = self
			.aliasable
			.entry(merge)
			.or_insert(Vec::new_in(self.resources.allocator()));

		for &i in merges.iter() {
			let res = &mut self.resources[i as usize];
			if res.try_merge(&mut resource) {
				self.resource_map.push(i);
				return;
			}
		}

		merges.push(self.resources.len() as u32);
		self.push(resource.into());
	}

	fn add(&mut self, resource: VirtualResourceData<'graph>) {
		match resource.ty {
			VirtualResourceType::Data(_)
			| VirtualResourceType::UploadBuffer(_)
			| VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::External(_),
				..
			})
			| VirtualResourceType::Image(GpuData {
				desc: ImageType::External(_),
				..
			}) => self.push(resource.into()),
			VirtualResourceType::GpuBuffer(GpuData {
				desc: GpuBufferType::Internal(_),
				..
			}) => unsafe { self.merge(MergeCandidate::GpuBuffer, resource) },
			VirtualResourceType::Image(GpuData {
				desc: ImageType::Internal(desc),
				..
			}) => unsafe { self.merge(MergeCandidate::Image(desc), resource) },
		}
	}

	fn finish(self, device: &Device, graph: &mut RenderGraph) -> ResourceMap<'graph> {
		let alloc = *self.resources.allocator();
		let dummy_buffer_sync = ExternalSync {
			semaphore: Semaphore::null(),
			value: 0,
			usage: Vec::new_in(alloc),
		};
		let dummy_image_sync = ExternalSync {
			semaphore: Semaphore::null(),
			value: 0,
			usage: Vec::new_in(alloc),
		};
		let mut buffers = Vec::new_in(alloc);
		let mut images = Vec::new_in(alloc);

		let resources = self.resources.into_iter().enumerate().map(|(i, desc)| match desc.ty {
			ResourceDescType::Data(data) => Resource::Data(data, DataState::Uninit),
			ResourceDescType::UploadBuffer(desc) => Resource::UploadBuffer(
				graph.caches.upload_buffers[graph.curr_frame]
					.get(device, desc)
					.expect("failed to allocate upload buffer"),
			),
			ResourceDescType::GpuBuffer(desc) => {
				buffers.push(i as _);
				Resource::GpuBuffer(BufferResource {
					resource: GpuResource {
						handle: graph
							.caches
							.gpu_buffers
							.get(device, desc.handle)
							.expect("failed to allocate gpu buffer"),
						usages: desc.usages,
					},
					prev_usage: dummy_buffer_sync.clone(),
					next_usage: dummy_buffer_sync.clone(),
				})
			},
			ResourceDescType::ExternalBuffer(buffer) => {
				buffers.push(i as _);
				Resource::GpuBuffer(buffer)
			},
			ResourceDescType::Image(desc) => {
				images.push(i as _);
				Resource::Image(SyncedResource {
					resource: GpuResource {
						handle: graph
							.caches
							.images
							.get(device, desc.handle)
							.expect("failed to allocate image"),
						usages: desc.usages,
					},
					prev_usage: dummy_image_sync.clone(),
					next_usage: dummy_image_sync.clone(),
				})
			},
			ResourceDescType::ExternalImage(image) => {
				images.push(i as _);
				Resource::Image(image)
			},
		});

		unsafe { ResourceMap::new(self.resource_map, resources.collect_in(alloc), buffers, images) }
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
	aspect: ImageAspectFlags,
}

#[derive(Clone)]
struct InProgressDependencyInfo<'graph> {
	barriers: ArenaMap<'graph, SyncPair<PipelineStageFlags2>, SyncPair<AccessFlags2>>,
	image_barriers: ArenaMap<'graph, Image, InProgressImageBarrier>,
}

#[derive(Clone)]
pub struct InProgressCrossQueueSync<'graph> {
	signal_semaphores: Vec<SemaphoreSubmitInfo, &'graph Arena>,
	signal_barriers: InProgressDependencyInfo<'graph>,
	wait_semaphores: Vec<SemaphoreSubmitInfo, &'graph Arena>,
	wait_barriers: InProgressDependencyInfo<'graph>,
}

#[derive(Clone)]
struct InProgressSync<'graph> {
	queue: InProgressDependencyInfo<'graph>,
	cross_queue: InProgressCrossQueueSync<'graph>,
}

impl<'graph> From<InProgressDependencyInfo<'graph>> for DependencyInfo<'graph> {
	fn from(b: InProgressDependencyInfo<'graph>) -> Self {
		let arena = *b.barriers.allocator();
		Self {
			barriers: b
				.barriers
				.into_iter()
				.map(|(s, a)| {
					GlobalBarrierAccess {
						previous_access: AccessInfo {
							stage_mask: s.from,
							access_mask: a.from,
							image_layout: ImageLayout::UNDEFINED,
						},
						next_access: AccessInfo {
							stage_mask: s.to,
							access_mask: a.to,
							image_layout: ImageLayout::UNDEFINED,
						},
					}
					.into()
				})
				.collect_in(arena),
			image_barriers: b
				.image_barriers
				.into_iter()
				.map(|(i, b)| {
					ImageBarrierAccess {
						previous_access: b.sync.from,
						next_access: b.sync.to,
						image: i,
						range: ImageSubresourceRange {
							aspect_mask: b.aspect,
							base_array_layer: 0,
							layer_count: REMAINING_ARRAY_LAYERS,
							base_mip_level: 0,
							level_count: REMAINING_MIP_LEVELS,
						},
						..Default::default()
					}
					.into()
				})
				.collect_in(arena),
		}
	}
}

impl<'graph> From<InProgressCrossQueueSync<'graph>> for CrossQueueSync<'graph> {
	fn from(s: InProgressCrossQueueSync<'graph>) -> Self {
		let arena = *s.signal_semaphores.allocator();
		Self {
			signal_semaphores: s.signal_semaphores.into_iter().map(Into::into).collect_in(arena),
			signal_barriers: s.signal_barriers.into(),
			wait_semaphores: s.wait_semaphores.into_iter().map(Into::into).collect_in(arena),
			wait_barriers: s.wait_barriers.into(),
		}
	}
}

impl<'graph> From<InProgressSync<'graph>> for Sync<'graph> {
	fn from(s: InProgressSync<'graph>) -> Self {
		let arena = *s.queue.barriers.allocator();
		Self {
			queue: QueueSync {
				barriers: s.queue.into(),
				set_events: Vec::new_in(arena),
				wait_events: Vec::new_in(arena),
			},
			cross_queue: s.cross_queue.into(),
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
	fn new<'a>(arena: &'graph Arena, passes: &'a [PassData<'a, 'a>]) -> Self {
		Self {
			sync: std::iter::repeat(InProgressSync {
				queue: InProgressDependencyInfo::default(arena),
				cross_queue: InProgressCrossQueueSync {
					signal_semaphores: Vec::new_in(arena),
					signal_barriers: InProgressDependencyInfo::default(arena),
					wait_semaphores: Vec::new_in(arena),
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
		&mut self, image: Image, aspect: ImageAspectFlags, prev_pass: u32, prev_access: AccessInfo, next_pass: u32,
		next_access: AccessInfo,
	) {
		let dep_info = self.get_dep_info(prev_pass, next_pass);
		Self::insert_info(dep_info, image, aspect, prev_access, next_access);
	}

	/// Wait for some external sync.
	///
	/// If a global barrier is required, pass `Image::null()` and `ImageAspectFlags::empty()`. If no layout transition
	/// is required, an image barrier will be converted to a global barrier.
	fn wait_external_sync(
		&mut self, image: Image, aspect: ImageAspectFlags, pass: u32, prev_sync: ExternalSync<AccessInfo>,
		next_access: AccessInfo,
	) {
		let pass = Self::before_pass(pass);
		let cross_queue = &mut self.sync[pass as usize].cross_queue;
		if prev_sync.usage.stage_mask != PipelineStageFlags2::NONE
			|| prev_sync.usage.access_mask != AccessFlags2::NONE
			|| prev_sync.usage.image_layout != ImageLayout::UNDEFINED
			|| next_access.image_layout != ImageLayout::UNDEFINED
		{
			Self::insert_info(
				&mut cross_queue.wait_barriers,
				image,
				aspect,
				prev_sync.usage,
				next_access,
			);
		}

		if prev_sync.semaphore != Semaphore::null() {
			cross_queue.wait_semaphores.push(
				SemaphoreSubmitInfo::builder()
					.semaphore(prev_sync.semaphore)
					.value(prev_sync.value)
					.stage_mask(next_access.stage_mask)
					.build(),
			);
		}
	}

	/// Signal some external sync.
	///
	/// If a global barrier is required, pass `Image::null()` and `ImageAspectFlags::empty()`. If no layout transition
	/// is required, an image barrier will be converted to a global barrier.
	fn signal_external_sync(
		&mut self, image: Image, aspect: ImageAspectFlags, pass: u32, prev_access: AccessInfo,
		next_sync: ExternalSync<AccessInfo>,
	) {
		let pass = Self::after_pass(pass);
		let cross_queue = &mut self.sync[pass as usize].cross_queue;
		if next_sync.usage.stage_mask != PipelineStageFlags2::NONE
			|| next_sync.usage.access_mask != AccessFlags2::NONE
			|| prev_access.image_layout != ImageLayout::UNDEFINED
			|| next_sync.usage.image_layout != ImageLayout::UNDEFINED
		{
			Self::insert_info(
				&mut cross_queue.signal_barriers,
				image,
				aspect,
				prev_access,
				next_sync.usage,
			);
		}

		if next_sync.semaphore != Semaphore::null() {
			cross_queue.signal_semaphores.push(
				SemaphoreSubmitInfo::builder()
					.semaphore(next_sync.semaphore)
					.value(next_sync.value)
					.stage_mask(prev_access.stage_mask)
					.build(),
			);
		}
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

	fn insert_info(
		dep_info: &mut InProgressDependencyInfo<'graph>, image: Image, aspect: ImageAspectFlags,
		prev_access: AccessInfo, mut next_access: AccessInfo,
	) {
		if next_access.stage_mask == PipelineStageFlags2::empty() {
			// Ensure that the stage masks are valid if no stages were determined
			next_access.stage_mask = PipelineStageFlags2::ALL_COMMANDS;
		}

		// Don't need to check if `image` is null, because `AccessInfo::image_layout` will always be undefined if a
		// global barrier is required.
		if prev_access.image_layout == next_access.image_layout {
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
			debug_assert!(image != Image::null(), "Image layout transition without an image");
			let old = dep_info.image_barriers.insert(
				image,
				InProgressImageBarrier {
					sync: SyncPair {
						from: prev_access,
						to: next_access,
					},
					aspect,
				},
			);
			debug_assert!(old.is_none(), "Multiple transitions for one image in a single pass");
		}
	}

	fn before_pass(pass: u32) -> u32 { pass }

	fn after_pass(pass: u32) -> u32 { pass + 1 }

	fn finish(
		self, device: &Device, event_list: &mut ResourceList<resource::Event>,
	) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let arena = *self.sync.allocator();
		let mut sync: Vec<Sync, _> = self.sync.into_iter().map(Into::into).collect_in(arena);

		for (pair, info) in self.events {
			let event = event_list.get_or_create(device, ())?;
			let info: DependencyInfo = info.into();

			sync[pair.from as usize].queue.set_events.push(EventInfo {
				event,
				info: info.clone(),
			});
			sync[pair.to as usize].queue.wait_events.push(EventInfo { event, info });
		}

		Ok(sync)
	}
}

impl<I: Copy + Into<UsageType>, A: std::alloc::Allocator> ExternalSync<Vec<I, A>> {
	fn as_prev(&self, discard: bool) -> ExternalSync<AccessInfo> {
		self.map(|usages| as_previous_access(usages.iter().map(|&x| x.into()), discard))
	}

	fn as_next(&self, prev_access: AccessInfo) -> ExternalSync<AccessInfo> {
		self.map(|usages| as_next_access(usages.iter().map(|&x| x.into()), prev_access))
	}
}

impl BufferUsageOwned<'_> {
	fn as_prev(&self) -> AccessInfo { as_previous_access(self.usages.iter().map(|&x| x.into()), false) }

	fn as_next(&self, prev_access: AccessInfo) -> AccessInfo {
		as_next_access(self.usages.iter().map(|&x| x.into()), prev_access)
	}

	fn is_write(&self) -> bool { self.usages.iter().any(|&x| is_write_access(x.into())) }
}

impl ImageUsageOwned<'_> {
	fn as_prev(&self) -> AccessInfo { as_previous_access(self.usages.iter().map(|&x| x.into()), false) }

	fn as_next(&self, prev_access: AccessInfo) -> AccessInfo {
		as_next_access(self.usages.iter().map(|&x| x.into()), prev_access)
	}

	// Returns (is_write, is_only_write)
	fn is_write(&self) -> (bool, bool) {
		let mut is_write = false;
		let mut is_only_write = true;
		for usage in self.usages.iter().map(|&x| x.into_usage()) {
			if is_write_access(usage) {
				is_write = true;
			} else {
				is_only_write = false;
			}
		}
		(is_write, is_only_write)
	}
}

struct Synchronizer<'temp, 'graph> {
	resource_map: &'temp ResourceMap<'graph>,
	passes: &'temp [PassData<'temp, 'graph>],
}

impl<'temp, 'graph> Synchronizer<'temp, 'graph> {
	fn new(resource_map: &'temp ResourceMap<'graph>, passes: &'temp [PassData<'temp, 'graph>]) -> Self {
		Self { resource_map, passes }
	}

	fn sync(
		&mut self, device: &Device, event_list: &mut ResourceList<resource::Event>,
	) -> Result<Vec<Sync<'graph>, &'graph Arena>> {
		let mut sync = SyncBuilder::new(self.resource_map.arena(), self.passes);

		for buffer in self.resource_map.buffers() {
			let mut usages = buffer.resource.usages.iter();
			let (mut prev_pass, mut prev_access) = {
				let (&pass, usage) = usages.next().unwrap();

				let prev_sync = buffer.prev_usage.as_prev(false);
				let next_access = usage.as_next(prev_sync.usage);
				sync.wait_external_sync(Image::null(), ImageAspectFlags::empty(), pass, prev_sync, next_access);

				let prev_access = usage.as_prev();
				let next_sync = buffer.next_usage.as_next(prev_access);
				sync.signal_external_sync(Image::null(), ImageAspectFlags::empty(), pass, prev_access, next_sync);

				debug_assert!(
					sync.sync[SyncBuilder::after_pass(pass) as usize]
						.cross_queue
						.signal_barriers
						.barriers
						.is_empty() || usages.len() == 0,
					"Cannot have signalling `ExternalSync` that is also used as an input to other passes."
				);

				(pass, prev_access)
			};

			let mut usages = usages.peekable();
			while let Some((&pass, usage)) = usages.next() {
				let next_pass = pass;
				let mut next_access = usage.as_next(prev_access);
				let mut next_prev_access = usage.as_prev();

				if usage.is_write() {
					// We're a write, so we can't merge future passes into us.
					sync.barrier(
						Image::null(),
						ImageAspectFlags::empty(),
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
					while let Some((&pass, usage)) = usages.peek() {
						if usage.is_write() {
							// We've hit a write, stop merging now.
							break;
						}
						last_read_pass = pass;
						next_access |= usage.as_next(prev_access);
						next_prev_access |= usage.as_prev();

						usages.next();
					}

					sync.barrier(
						Image::null(),
						ImageAspectFlags::empty(),
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

		for image in self.resource_map.images() {
			let mut usages = image.resource.usages.iter();
			let (mut prev_pass, mut prev_access) = {
				let (&pass, usage) = usages.next().unwrap();

				let (_, is_only_write) = usage.is_write();
				let prev_sync = image.prev_usage.as_prev(is_only_write);
				let next_access = usage.as_next(prev_sync.usage);
				sync.wait_external_sync(image.resource.handle, usage.aspect, pass, prev_sync, next_access);

				let prev_access = usage.as_prev();
				let next_sync = image.next_usage.as_next(prev_access);
				sync.signal_external_sync(image.resource.handle, usage.aspect, pass, prev_access, next_sync);

				debug_assert!(
					sync.sync[SyncBuilder::after_pass(pass) as usize]
						.cross_queue
						.signal_barriers
						.barriers
						.is_empty() || usages.len() == 0,
					"Cannot have signalling `ExternalSync` that is also used as an input to other passes."
				);

				(pass, prev_access)
			};

			let mut usages = usages.peekable();
			while let Some((&pass, usage)) = usages.next() {
				let next_pass = pass;
				let mut next_access = usage.as_next(prev_access);
				let mut next_prev_access = usage.as_prev();

				let (is_write, is_only_write) = usage.is_write();
				if is_write {
					// We're a write, so we can't merge future passes into us.
					if is_only_write {
						// If this pass will only write to the image, don't preserve it's contents.
						prev_access.image_layout = ImageLayout::UNDEFINED;
					}
					sync.barrier(
						image.resource.handle,
						usage.aspect,
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
					let mut aspect = usage.aspect;
					while let Some((&pass, usage)) = usages.peek() {
						let as_next = usage.as_next(prev_access);
						if usage.is_write().0 || as_next.image_layout != prev_access.image_layout {
							// We've hit a write, stop merging now.
							// Note that read -> read layout transitions are also writes.
							break;
						}
						last_read_pass = pass;
						next_access |= as_next;
						next_prev_access |= usage.as_prev();
						aspect |= usage.aspect;

						usages.next();
					}

					sync.barrier(
						image.resource.handle,
						aspect,
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
				.fold(ResourceAliaser::new(self.arena), |mut a, r| {
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
