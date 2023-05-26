use std::{collections::VecDeque, io::Write, ops::Range};

use ash::vk;
use radiance_graph::{
	arena::Arena,
	cmd::CommandPool,
	device::{Device, QueueType, Queues},
	gpu_allocator::MemoryLocation,
	graph::{ImageUsageType, SemaphoreInfo, TimelineSemaphore},
	resource::{Buffer, BufferDesc},
	sync::{as_next_access, as_previous_access},
	Result,
};

pub struct Staging {
	inner: CircularBuffer,
	semaphore: TimelineSemaphore,
	pools: Queues<CommandPool>,
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

impl Staging {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			inner: CircularBuffer::new(device)?,
			semaphore: TimelineSemaphore::new(device)?,
			pools: device
				.queue_families()
				.try_map_ref(|&queue| CommandPool::new(device, queue))?,
		})
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.semaphore.wait(device).unwrap();

		self.inner.destroy(device);
		self.semaphore.destroy(device);
		self.pools.map(|x| x.destroy(device));
	}

	/// Stage some GPU resources.
	///
	/// `wait` will be waited upon before the staging commands are submitted.
	///
	/// The returned `StageTicket` can be used to wait for the staging to complete.
	pub fn stage(
		&mut self, device: &Device, mut wait: Vec<vk::SemaphoreSubmitInfo, &Arena>,
		exec: impl FnOnce(&mut StagingCtx) -> Result<()>,
	) -> Result<StageTicket> {
		self.inner.for_submit(|inner| {
			let mut ctx = StagingCtx {
				device,
				inner,
				pre_bufs: self.pools.map_ref(|_| None),
				post_bufs: self.pools.map_ref(|_| None),
				queues: &mut self.pools,
			};
			exec(&mut ctx)?;

			submit_queues(device, &mut self.semaphore, ctx.pre_bufs, &mut wait)?;
			submit_queues(device, &mut self.semaphore, ctx.post_bufs, &mut wait)?;

			Ok(self.semaphore.value())
		})?;

		let semaphore = self.semaphore.semaphore();
		let value = self.semaphore.value();

		Ok(StageTicket { semaphore, value })
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
			}
		}

		Ok(())
	}
}

pub struct StagingCtx<'a> {
	device: &'a Device,
	inner: &'a mut CircularBuffer,
	queues: &'a mut Queues<CommandPool>,
	pre_bufs: Queues<Option<vk::CommandBuffer>>,
	post_bufs: Queues<Option<vk::CommandBuffer>>,
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
	/// Copy data from CPU memory to a GPU buffer.
	pub fn stage_buffer(&mut self, data: &[u8], dst: vk::Buffer, dst_offset: u64) -> Result<()> {
		let loc = self.inner.copy(self.device, data)?;
		let buf = self.pre_buf(QueueType::Transfer)?;
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
	pub fn stage_image<'a>(
		&mut self, data: &[u8], image: vk::Image, region: ImageStage, discard: bool, queue: QueueType,
		prev_usages: &[ImageUsageType], next_usages: &[ImageUsageType],
	) -> Result<()> {
		let loc = self.inner.copy(self.device, data)?;
		unsafe {
			let old_queue = *self.device.queue_families().get(queue);
			let new_queue = *self.device.queue_families().get(QueueType::Transfer);

			let range = vk::ImageSubresourceRange {
				aspect_mask: region.image_subresource.aspect_mask,
				base_mip_level: region.image_subresource.mip_level,
				level_count: 1,
				base_array_layer: region.image_subresource.base_array_layer,
				layer_count: region.image_subresource.layer_count,
			};
			let tbuf = self.pre_buf(QueueType::Transfer)?;

			// `queue` to transfer QFOT.
			let prev_access = as_previous_access(prev_usages.iter().map(|&x| x.into()), discard);
			let discard = prev_access.image_layout == vk::ImageLayout::UNDEFINED;
			let old_layout = if !discard {
				// The release barrier is only required if the contents are to be preserved
				self.device.device().cmd_pipeline_barrier2(
					self.pre_buf(queue)?,
					&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
						.image(image)
						.subresource_range(range)
						.old_layout(prev_access.image_layout)
						.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
						.src_access_mask(prev_access.access_mask)
						.src_stage_mask(prev_access.stage_mask)
						.build()]),
				);
				prev_access.image_layout
			} else {
				vk::ImageLayout::UNDEFINED
			};
			let barr = vk::ImageMemoryBarrier2::builder()
				.image(image)
				.subresource_range(range)
				.old_layout(old_layout)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
				.dst_stage_mask(vk::PipelineStageFlags2::TRANSFER);
			self.device.device().cmd_pipeline_barrier2(
				tbuf,
				&vk::DependencyInfo::builder().image_memory_barriers(&[if !discard {
					barr.src_queue_family_index(old_queue)
						.dst_queue_family_index(new_queue)
						.build()
				} else {
					barr.build()
				}]),
			);

			self.device.device().cmd_copy_buffer_to_image(
				tbuf,
				loc.buffer,
				image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[vk::BufferImageCopy {
					buffer_offset: loc.offset as u64,
					buffer_row_length: region.buffer_row_length,
					buffer_image_height: region.buffer_image_height,
					image_subresource: region.image_subresource,
					image_offset: region.image_offset,
					image_extent: region.image_extent,
				}],
			);

			// transfer to `queue` QFOT.
			let next_access = as_next_access(next_usages.iter().map(|&x| x.into()), prev_access);
			self.device.device().cmd_pipeline_barrier2(
				tbuf,
				&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
					.image(image)
					.subresource_range(range)
					.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.new_layout(next_access.image_layout)
					.src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
					.src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
					.src_queue_family_index(new_queue)
					.dst_queue_family_index(old_queue)
					.build()]),
			);
			self.device.device().cmd_pipeline_barrier2(
				self.post_buf(queue)?,
				&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
					.image(image)
					.subresource_range(range)
					.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.new_layout(next_access.image_layout)
					.dst_access_mask(next_access.access_mask)
					.dst_stage_mask(next_access.stage_mask)
					.src_queue_family_index(new_queue)
					.dst_queue_family_index(old_queue)
					.build()]),
			);

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

#[derive(Copy, Clone)]
struct BufferLoc<B> {
	buffer: B,
	offset: usize,
}

impl CircularBuffer {
	/// 4 MB.
	const BUFFER_SIZE: usize = 1024 * 1024 * 4;

	fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			buffers: vec![Buffer::create(
				device,
				BufferDesc {
					size: Self::BUFFER_SIZE,
					usage: vk::BufferUsageFlags::TRANSFER_SRC,
				},
				MemoryLocation::GpuToCpu, // This guarantees that we won't use up the 256 MB of BAR we have.
			)?],
			head: BufferLoc { buffer: 0, offset: 0 },
			tail: BufferLoc { buffer: 0, offset: 0 },
			submits: VecDeque::new(),
		})
	}

	unsafe fn destroy(self, device: &Device) {
		for buffer in self.buffers {
			buffer.destroy(device);
		}
	}

	// Returns the index of the submit in the list.
	fn for_submit(&mut self, exec: impl FnOnce(&mut Self) -> Result<u64>) -> Result<()> {
		let mut range = self.tail..self.tail;
		let sem_value = exec(self)?;
		range.end = self.tail;

		self.submits.push_back(SubmitInfo { range, sem_value });

		Ok(())
	}

	fn copy(&mut self, device: &Device, data: &[u8]) -> Result<BufferLoc<vk::Buffer>> {
		let size = data.len();

		// Ensure we have enough space after the tail pointer.
		if self.buffers[self.tail.buffer].size() as usize - self.tail.offset < size {
			// We don't have enough space, wrap around.
			self.tail.buffer = (self.tail.buffer + 1) % self.buffers.len();
			self.tail.offset = 0;
			if self.tail.buffer == self.head.buffer {
				// We've wrapped back to the head, so we need to allocate a new buffer.
				self.insert_buffer(device, size)?;
			}
		}

		let buffer = &mut self.buffers[self.tail.buffer];
		unsafe {
			let mut slice = &mut buffer.mapped_ptr().unwrap().as_mut()[self.tail.offset..];
			slice.write(data).unwrap();
		}

		self.tail.offset += size;

		Ok(BufferLoc {
			buffer: buffer.inner(),
			offset: self.tail.offset - size,
		})
	}

	fn insert_buffer(&mut self, device: &Device, size: usize) -> Result<()> {
		self.buffers.insert(
			self.tail.buffer,
			Buffer::create(
				device,
				BufferDesc {
					size: Self::BUFFER_SIZE.max(size),
					usage: vk::BufferUsageFlags::TRANSFER_SRC,
				},
				MemoryLocation::GpuToCpu,
			)?,
		);

		// Increment the head because `insert` just shifted it.
		self.head.buffer += 1;
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
	match queues {
		Queues::Separate {
			graphics,
			compute,
			transfer,
		} => {
			if let Some(graphics) = graphics {
				submit(device, semaphore, QueueType::Graphics, graphics, wait)?;
			}

			if let Some(compute) = compute {
				submit(device, semaphore, QueueType::Compute, compute, wait)?;
			}

			if let Some(transfer) = transfer {
				submit(device, semaphore, QueueType::Transfer, transfer, wait)?;
			}
		},
		Queues::Single(queue) => {
			if let Some(queue) = queue {
				submit(device, semaphore, QueueType::Graphics, queue, wait)?;
			}
		},
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
				.wait_semaphore_infos(&wait)
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
