use std::{collections::VecDeque, io::Write, ops::Range};

use ash::vk;
use radiance_graph::{
	arena::Arena,
	device::{cmd::CommandPool, Device},
	gpu_allocator::MemoryLocation,
	graph::TimelineSemaphore,
	resource::{Buffer, BufferDesc},
	Result,
};

pub struct Staging {
	inner: CircularBuffer,
	semaphore: TimelineSemaphore,
	pool: CommandPool,
}

pub struct StageTicket {
	index: usize,
	value: u64,
}

impl Staging {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			inner: CircularBuffer::new(device)?,
			semaphore: TimelineSemaphore::new(device)?,
			pool: CommandPool::new(device, *device.queue_families().transfer())?,
		})
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.inner.destroy(device);
		self.semaphore.destroy(device);
		self.pool.destroy(device);
	}

	/// Stage some GPU resources.
	///
	/// `semaphores` will be waited upon before the staging commands are submitted.
	pub fn stage(
		&mut self, device: &Device, mut semaphores: Vec<vk::SemaphoreSubmitInfo, &Arena>,
		exec: impl FnOnce(&mut StagingCtx) -> Result<()>,
	) -> Result<StageTicket> {
		let buf = self.pool.next(device)?;
		unsafe {
			device.device().begin_command_buffer(
				buf,
				&vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)?;
		}

		let mut needs_wait = false;
		let index = self.inner.for_submit(|inner| {
			let mut ctx = StagingCtx {
				device,
				inner,
				buf,
				sem: &mut self.semaphore,
				needs_qfot: false,
			};
			exec(&mut ctx)?;
			needs_wait = ctx.needs_qfot;

			let (_, value) = ctx.sem.next();
			Ok(value)
		})?;

		let sem = self.semaphore.semaphore();
		let value = self.semaphore.value();

		unsafe {
			device.device().end_command_buffer(buf)?;

			if needs_wait {
				semaphores.push(
					vk::SemaphoreSubmitInfo::builder()
						.semaphore(sem)
						.value(value - 1)
						.stage_mask(vk::PipelineStageFlags2::TRANSFER)
						.build(),
				);
			}
			device.submit_transfer(
				&[vk::SubmitInfo2::builder()
					.wait_semaphore_infos(&semaphores)
					.command_buffer_infos(&[vk::CommandBufferSubmitInfo::builder().command_buffer(buf).build()])
					.signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::builder()
						.semaphore(sem)
						.value(value)
						.stage_mask(vk::PipelineStageFlags2::TRANSFER)
						.build()])
					.build()],
				vk::Fence::null(),
			)?;
		}

		Ok(StageTicket { index, value })
	}

	/// Poll and reclaim any buffer space that is no longer in use.
	/// Returns `true` if the staging was reset.
	pub fn poll(&mut self, device: &Device) -> Result<bool> {
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
				self.pool.reset(device)?;
				Ok(true)
			}
		} else {
			Ok(false)
		}
	}
}

pub struct StagingCtx<'a> {
	device: &'a Device,
	inner: &'a mut CircularBuffer,
	buf: vk::CommandBuffer,
	needs_qfot: bool,
	sem: &'a mut TimelineSemaphore,
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
	///
	/// `next_usages` gives the next usages of the buffer after the transfer. Appropriate synchronization is performed
	/// against these usages.
	pub fn stage_buffer(&mut self, data: &[u8], dst: vk::Buffer, dst_offset: u64) -> Result<()> {
		let loc = self.inner.copy(self.device, data)?;
		unsafe {
			self.device.device().cmd_copy_buffer(
				self.buf,
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
	/// `next_usages` gives the next usages of the image after the transfer.
	///
	/// You will have to manually QFOT the image from `old_queue`, and submit the barriers before the [`Staging::stage`]
	/// closure returns. You will also have to QFOT the image back to the original queue after the call to
	/// `Staging::stage`.
	///
	/// This is more complicated than staging a buffer because images are always exclusively owned by a queue.
	pub fn stage_image<'a>(
		&mut self, data: &[u8], image: vk::Image, region: ImageStage, old_queue: u32, old_layout: vk::ImageLayout,
		new_layout: vk::ImageLayout,
	) -> Result<()> {
		if !self.needs_qfot {
			self.needs_qfot = true;
			self.sem.next(); // Signalled once by the caller's QFOTs
		}

		let loc = self.inner.copy(self.device, data)?;
		unsafe {
			let (src_queue, dst_queue) = if self.device.needs_queue_ownership_transfer() {
				(old_queue, *self.device.queue_families().transfer())
			} else {
				(0, 0)
			};

			let range = vk::ImageSubresourceRange {
				aspect_mask: region.image_subresource.aspect_mask,
				base_mip_level: region.image_subresource.mip_level,
				level_count: 1,
				base_array_layer: region.image_subresource.base_array_layer,
				layer_count: region.image_subresource.layer_count,
			};

			self.device.device().cmd_pipeline_barrier2(
				self.buf,
				&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
					.image(image)
					.subresource_range(range)
					.old_layout(if region.image_offset == vk::Offset3D::default() {
						vk::ImageLayout::UNDEFINED
					} else {
						old_layout
					})
					.src_queue_family_index(src_queue)
					.dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
					.dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
					.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.dst_queue_family_index(dst_queue)
					.build()]),
			);
			self.device.device().cmd_copy_buffer_to_image(
				self.buf,
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
			self.device.device().cmd_pipeline_barrier2(
				self.buf,
				&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
					.image(image)
					.subresource_range(range)
					.src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
					.src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
					.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.src_queue_family_index(dst_queue)
					.new_layout(new_layout)
					.dst_queue_family_index(src_queue)
					.build()]),
			);

			Ok(())
		}
	}

	/// Get the semaphore and value to signal after all source to transfer QFOT barriers are executed.
	pub fn signal_semaphore(&self) -> Option<(vk::Semaphore, u64)> {
		if self.needs_qfot {
			Some((self.sem.semaphore(), self.sem.value()))
		} else {
			None
		}
	}

	/// Get the semaphore and value to wait on before all transfer to source QFOT barriers are executed.
	pub fn wait_semaphore(&self) -> Option<(vk::Semaphore, u64)> {
		if self.needs_qfot {
			Some((self.sem.semaphore(), self.sem.value() + 1))
		} else {
			None
		}
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
	fn for_submit(&mut self, exec: impl FnOnce(&mut Self) -> Result<u64>) -> Result<usize> {
		let mut range = self.tail..self.tail;
		let sem_value = exec(self)?;
		range.end = self.tail;

		let ret = self.submits.len();
		self.submits.push_back(SubmitInfo { range, sem_value });

		Ok(ret)
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
