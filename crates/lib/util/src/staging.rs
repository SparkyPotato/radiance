use std::{collections::VecDeque, fmt::Debug, io::Write, ops::Range};

use ash::vk;
use radiance_graph::{
	arena::Arena,
	cmd::CommandPool,
	device::{Device, QueueType, Queues},
	graph::{ImageUsageType, SemaphoreInfo, TimelineSemaphore},
	resource::{Buffer, BufferDesc},
	sync::{as_next_access, as_previous_access, get_access_info, ImageBarrierAccess, UsageType},
	MemoryLocation,
	Result,
};

pub struct Staging {
	inner: CircularBuffer,
	semaphore: TimelineSemaphore,
	pools: Queues<CommandPool>,
	min_granularity: vk::Extent3D,
}

pub struct StageTicket {
	semaphore: vk::Semaphore,
	value: u64,
}

impl StageTicket {
	pub fn as_info(&self) -> SemaphoreInfo {
		SemaphoreInfo {
			semaphore: self.semaphore,
			value: self.value,
			stage: vk::PipelineStageFlags2::ALL_COMMANDS,
		}
	}
}

pub enum StageError<E> {
	User(E),
	Vulkan(radiance_graph::Error),
}

impl<E> From<E> for StageError<E> {
	fn from(e: E) -> Self { StageError::User(e) }
}

impl<E: Debug> Debug for StageError<E> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			StageError::User(e) => write!(f, "User({:?})", e),
			StageError::Vulkan(e) => write!(f, "Vulkan({:?})", e),
		}
	}
}

impl Staging {
	pub fn new(device: &Device) -> Result<Self> {
		let min_granularity = unsafe {
			let props = device
				.instance()
				.get_physical_device_queue_family_properties(device.physical_device());
			props[device.queue_families().transfer as usize].min_image_transfer_granularity
		};

		Ok(Self {
			inner: CircularBuffer::new(device)?,
			semaphore: TimelineSemaphore::new(device)?,
			pools: device
				.queue_families()
				.try_map_ref(|&queue| CommandPool::new(device, queue))?,
			min_granularity,
		})
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			self.semaphore.wait(device).unwrap();

			self.inner.destroy(device);
			self.semaphore.destroy(device);
			self.pools.map(|x| x.destroy(device));
		}
	}

	/// Stage some GPU resources.
	///
	/// `wait` will be waited upon before the staging commands are submitted.
	///
	/// The returned `StageTicket` can be used to wait for the staging to complete.
	pub fn stage<T, E>(
		&mut self, device: &Device, mut wait: Vec<vk::SemaphoreSubmitInfo, &Arena>,
		exec: impl FnOnce(&mut StagingCtx) -> std::result::Result<T, E>,
	) -> std::result::Result<(T, StageTicket), StageError<E>> {
		let ret = self.inner.for_submit(|inner| {
			let mut ctx = StagingCtx {
				device,
				inner,
				pre_bufs: self.pools.map_ref(|_| None),
				post_bufs: self.pools.map_ref(|_| None),
				queues: &mut self.pools,
				min_granularity: self.min_granularity,
			};
			let ret = exec(&mut ctx).map_err(|x| StageError::User(x))?;

			submit_queues(device, &mut self.semaphore, ctx.pre_bufs, &mut wait).map_err(|x| StageError::Vulkan(x))?;
			submit_queues(device, &mut self.semaphore, ctx.post_bufs, &mut wait).map_err(|x| StageError::Vulkan(x))?;

			Ok::<_, StageError<E>>((ret, self.semaphore.value()))
		})?;

		let semaphore = self.semaphore.semaphore();
		let value = self.semaphore.value();

		Ok((ret, StageTicket { semaphore, value }))
	}

	/// Poll and reclaim any buffer space that is no longer in use.
	pub fn poll(&mut self, device: &Device) -> Result<()> {
		let value = unsafe {
			device
				.device()
				.get_semaphore_counter_value(self.semaphore.semaphore())?
		};

		self.inner.submits.retain(|submit| {
			if submit.sem_value > value {
				true
			} else {
				self.inner.head = submit.range.end;
				false
			}
		});

		if self.inner.submits.is_empty() {
			unsafe {
				self.pools.map_mut(|x| x.reset(device)).try_map(|x| x)?;
				self.inner.head = BufferLoc::default();
				self.inner.tail = BufferLoc::default();
			}
		}

		Ok(())
	}
}

pub struct StagingCtx<'d> {
	pub device: &'d Device,
	inner: &'d mut CircularBuffer,
	queues: &'d mut Queues<CommandPool>,
	pre_bufs: Queues<Option<vk::CommandBuffer>>,
	post_bufs: Queues<Option<vk::CommandBuffer>>,
	min_granularity: vk::Extent3D,
}

pub struct ImageStage {
	/// Zero implies tightly packed.
	pub buffer_row_length: u32,
	/// Zero implies tightly packed.
	pub buffer_image_height: u32,
	pub image_subresource: vk::ImageSubresourceLayers,
	pub image_offset: vk::Offset3D,
	pub image_extent: vk::Extent3D,
}

impl StagingCtx<'_> {
	/// Get a command buffer to execute some custom staging commands before the other staging commands.
	///
	/// Appropriate synchronization must be performed.
	pub unsafe fn execute_before(&mut self, ty: QueueType) -> Result<vk::CommandBuffer> { self.pre_buf(ty) }

	/// Get a command buffer to execute some custom staging commands with the other staging commands.
	/// The ordering will be the same as the function call order on `self`.
	/// Appropriate synchronization must be performed.
	pub unsafe fn execute_with(&mut self, ty: QueueType) -> Result<vk::CommandBuffer> { self.post_buf(ty) }

	/// Copy data from CPU memory to a GPU buffer.
	pub fn stage_buffer(&mut self, data: &[u8], dst: vk::Buffer, dst_offset: u64) -> Result<()> {
		let loc = self.inner.copy(self.device, data)?;
		let buf = self.post_buf(QueueType::Transfer)?;
		unsafe {
			self.device.device().cmd_copy_buffer(
				buf,
				loc.buffer,
				dst,
				&[vk::BufferCopy {
					src_offset: loc.offset as u64,
					dst_offset,
					size: data.len() as u64,
				}],
			);
		}

		Ok(())
	}

	/// Copy data from CPU memory to a GPU image.
	///
	/// - `discard` is whether the contents of the image before the copy will be discarded.
	/// - `queue` is the current queue type that owns the image. Ownership will be returned to the queue.
	#[allow(clippy::too_many_arguments)]
	pub fn stage_image(
		&mut self, data: &[u8], image: vk::Image, region: ImageStage, discard: bool, queue: QueueType,
		prev_usages: &[ImageUsageType], next_usages: &[ImageUsageType],
	) -> Result<()> {
		let loc = self.inner.copy(self.device, data)?;
		unsafe {
			let old_queue = *self.device.queue_families().get(queue);
			let new_queue = *self.device.queue_families().get(QueueType::Transfer);

			let previous_access = as_previous_access(prev_usages.iter().map(|&x| x.into()), discard);
			let transfer_access = get_access_info(UsageType::TransferWrite);
			let next_access = as_next_access(next_usages.iter().map(|&x| x.into()), previous_access);

			let range = vk::ImageSubresourceRange {
				aspect_mask: region.image_subresource.aspect_mask,
				base_mip_level: region.image_subresource.mip_level,
				level_count: 1,
				base_array_layer: region.image_subresource.base_array_layer,
				layer_count: region.image_subresource.layer_count,
			};
			let copy = vk::BufferImageCopy {
				buffer_offset: loc.offset as u64,
				buffer_row_length: region.buffer_row_length,
				buffer_image_height: region.buffer_image_height,
				image_subresource: region.image_subresource,
				image_offset: region.image_offset,
				image_extent: region.image_extent,
			};

			let to_transfer = ImageBarrierAccess {
				image,
				range,
				previous_access,
				next_access: transfer_access,
				src_queue_family_index: old_queue,
				dst_queue_family_index: new_queue,
			};
			let to_original = ImageBarrierAccess {
				image,
				range,
				previous_access: transfer_access,
				next_access,
				src_queue_family_index: new_queue,
				dst_queue_family_index: old_queue,
			};

			if old_queue == new_queue
				|| region.image_offset.x as u32 % self.min_granularity.width != 0
				|| region.image_offset.y as u32 % self.min_granularity.height != 0
				|| region.image_offset.z as u32 % self.min_granularity.depth != 0
			{
				// We can't use multiple queues.
				let buf = self.pre_buf(queue)?;
				self.device.device().cmd_pipeline_barrier2(
					buf,
					&vk::DependencyInfo::builder().image_memory_barriers(&[to_transfer.as_no_qfot_barrier().into()]),
				);
				self.device.device().cmd_copy_buffer_to_image(
					buf,
					loc.buffer,
					image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[copy],
				);
				self.device.device().cmd_pipeline_barrier2(
					buf,
					&vk::DependencyInfo::builder().image_memory_barriers(&[to_original.as_no_qfot_barrier().into()]),
				);
			} else {
				// `queue` to transfer QFOT.
				let discard = previous_access.image_layout == vk::ImageLayout::UNDEFINED;
				if !discard {
					// The release barrier is only required if the contents are to be preserved
					self.device.device().cmd_pipeline_barrier2(
						self.pre_buf(queue)?,
						&vk::DependencyInfo::builder()
							.image_memory_barriers(&[to_transfer.as_release_barrier().into()]),
					);
				}
				let tbuf = self.post_buf(QueueType::Transfer)?;
				let barr = to_transfer.as_acquire_barrier();
				self.device.device().cmd_pipeline_barrier2(
					tbuf,
					&vk::DependencyInfo::builder().image_memory_barriers(&[if !discard {
						barr.into()
					} else {
						ImageBarrierAccess {
							src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
							dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
							..barr
						}
						.into()
					}]),
				);
				self.device.device().cmd_copy_buffer_to_image(
					tbuf,
					loc.buffer,
					image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[copy],
				);
				// transfer to `queue` QFOT.
				self.device.device().cmd_pipeline_barrier2(
					tbuf,
					&vk::DependencyInfo::builder().image_memory_barriers(&[to_original.as_release_barrier().into()]),
				);
				self.device.device().cmd_pipeline_barrier2(
					self.post_buf(queue)?,
					&vk::DependencyInfo::builder().image_memory_barriers(&[to_original.as_acquire_barrier().into()]),
				);
			}

			Ok(())
		}
	}

	fn pre_buf(&mut self, ty: QueueType) -> Result<vk::CommandBuffer> {
		get_buf(self.device, &mut self.pre_bufs, self.queues, ty)
	}

	fn post_buf(&mut self, ty: QueueType) -> Result<vk::CommandBuffer> {
		get_buf(self.device, &mut self.post_bufs, self.queues, ty)
	}
}

#[derive(Clone)]
struct SubmitInfo {
	sem_value: u64,
	range: Range<BufferLoc<usize>>,
}

struct CircularBuffer {
	buffers: Vec<Buffer>,
	head: BufferLoc<usize>,
	tail: BufferLoc<usize>,
	submits: VecDeque<SubmitInfo>,
}

#[derive(Copy, Clone, Default, Debug)]
struct BufferLoc<B> {
	buffer: B,
	offset: usize,
}

// TODO: Keeps growing for some reason.
// Also might be a time when the tail grows back into the head, making the buffer full, but later
// allocations think it's empty.
impl CircularBuffer {
	/// 12 MB.
	const BUFFER_SIZE: u64 = 1024 * 1024 * 12;

	fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			buffers: vec![Buffer::create(
				device,
				BufferDesc {
					size: Self::BUFFER_SIZE,
					usage: vk::BufferUsageFlags::TRANSFER_SRC,
				},
				MemoryLocation::CpuToGpu,
			)?],
			head: BufferLoc::default(),
			tail: BufferLoc::default(),
			submits: VecDeque::new(),
		})
	}

	unsafe fn destroy(self, device: &Device) {
		for buffer in self.buffers {
			buffer.destroy(device);
		}
	}

	// Returns the index of the submit in the list.
	fn for_submit<T, E>(
		&mut self, exec: impl FnOnce(&mut Self) -> std::result::Result<(T, u64), E>,
	) -> std::result::Result<T, E> {
		let mut range = self.tail..self.tail;
		let (ret, sem_value) = exec(self)?;
		range.end = self.tail;

		self.submits.push_back(SubmitInfo { range, sem_value });

		Ok(ret)
	}

	fn check(&mut self) -> &mut [u8] {
		unsafe {
			if self.tail.buffer == self.head.buffer && self.tail.offset < self.head.offset {
				&mut self.buffers[self.tail.buffer].data().as_mut()[self.tail.offset..self.head.offset]
			} else {
				&mut self.buffers[self.tail.buffer].data().as_mut()[self.tail.offset..]
			}
		}
	}

	fn copy(&mut self, device: &Device, data: &[u8]) -> Result<BufferLoc<vk::Buffer>> {
		let size = data.len();

		// Try all buffers until we find some space.
		while self.tail.buffer != self.head.buffer && self.check().len() < size {
			self.tail.buffer = (self.tail.buffer + 1) % self.buffers.len();
			self.tail.offset = 0;
		}
		// We're back at the head, try another time.
		if self.check().len() < size {
			self.insert_buffer(device, size as u64)?;
		}

		self.check().write_all(data).unwrap();

		self.tail.offset += size;

		Ok(BufferLoc {
			buffer: self.buffers[self.tail.buffer].inner(),
			offset: self.tail.offset - size,
		})
	}

	fn insert_buffer(&mut self, device: &Device, size: u64) -> Result<()> {
		self.buffers.insert(
			self.tail.buffer,
			Buffer::create(
				device,
				BufferDesc {
					size: Self::BUFFER_SIZE.max(size),
					usage: vk::BufferUsageFlags::TRANSFER_SRC,
				},
				MemoryLocation::CpuToGpu,
			)?,
		);
		self.tail.offset = 0;

		// Increment the head because `insert` just shifted it.
		if self.head.buffer >= self.tail.buffer {
			self.head.buffer += 1
		}
		// Update the submit ranges.
		for info in self.submits.iter_mut() {
			if info.range.start.buffer >= self.tail.buffer {
				info.range.start.buffer += 1;
			}
			if info.range.end.buffer >= self.tail.buffer {
				info.range.end.buffer += 1;
			}
		}

		Ok(())
	}
}

fn get_buf(
	device: &Device, bufs: &mut Queues<Option<vk::CommandBuffer>>, queues: &mut Queues<CommandPool>, ty: QueueType,
) -> Result<vk::CommandBuffer> {
	let buf = bufs.get_mut(ty);
	if let Some(buf) = buf {
		Ok(*buf)
	} else {
		let b = queues.get_mut(ty).next(device)?;
		unsafe {
			device.device().begin_command_buffer(
				b,
				&vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)?;
		}
		*buf = Some(b);
		Ok(b)
	}
}

fn submit_queues(
	device: &Device, semaphore: &mut TimelineSemaphore, queues: Queues<Option<vk::CommandBuffer>>,
	wait: &mut Vec<vk::SemaphoreSubmitInfo, &Arena>,
) -> Result<()> {
	let Queues {
		graphics,
		compute,
		transfer,
	} = queues;
	if let Some(transfer) = transfer {
		submit(device, semaphore, QueueType::Transfer, transfer, wait)?;
	}

	if let Some(compute) = compute {
		submit(device, semaphore, QueueType::Compute, compute, wait)?;
	}

	if let Some(graphics) = graphics {
		submit(device, semaphore, QueueType::Graphics, graphics, wait)?;
	}

	Ok(())
}

fn submit(
	device: &Device, semaphore: &mut TimelineSemaphore, ty: QueueType, buf: vk::CommandBuffer,
	wait: &mut Vec<vk::SemaphoreSubmitInfo, &Arena>,
) -> Result<()> {
	unsafe {
		device.device().end_command_buffer(buf)?;

		let (sem, value) = semaphore.next();
		wait.push(
			vk::SemaphoreSubmitInfo::builder()
				.semaphore(sem)
				.value(value - 1)
				.stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
				.build(),
		);
		device.submit(
			ty,
			&[vk::SubmitInfo2::builder()
				.wait_semaphore_infos(wait)
				.command_buffer_infos(&[vk::CommandBufferSubmitInfo::builder().command_buffer(buf).build()])
				.signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::builder()
					.semaphore(sem)
					.value(value)
					.stage_mask(vk::PipelineStageFlags2::TRANSFER)
					.build()])
				.build()],
			vk::Fence::null(),
		)?;
		wait.clear();

		Ok(())
	}
}
