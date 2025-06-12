use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::Arena,
	cmd::CommandPool,
	device::{Device, Graphics, QueueWaitOwned, SyncPoint, SyncStage},
	graph::compile::{DependencyInfo, Sync},
	Result,
};

pub struct FrameData {
	sync: SyncPoint<Graphics>,
	pool: CommandPool,
}

impl FrameData {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			sync: SyncPoint::default(),
			pool: CommandPool::new(device, device.queue_families().into::<Graphics>())?,
		})
	}

	pub fn reset(&mut self, device: &Device) -> Result<()> {
		unsafe {
			let span = span!(Level::TRACE, "wait for gpu");
			let _e = span.enter();
			// Let GPU finish this frame before doing anything else.
			self.sync.wait(device)?;
			self.pool.reset(device)?;

			Ok(())
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		unsafe {
			// Let GPU finish this frame before doing anything else.
			let _ = self.sync.wait(device);
			self.pool.destroy(device);
		}
	}
}

pub struct Submitter<'a, I> {
	cached_wait: QueueWaitOwned<&'a Arena>,
	data: &'a mut FrameData,
	buf: vk::CommandBuffer,
	sync: I,
}

impl<'a, I: Iterator<Item = Sync<'a>>> Submitter<'a, I> {
	pub fn new(
		arena: &'a Arena, sync: impl IntoIterator<IntoIter = I>, frames: &'a mut [FrameData], curr_frame: usize,
	) -> Self {
		let point = frames[curr_frame ^ 1].sync;
		Self {
			cached_wait: QueueWaitOwned {
				graphics: Some(SyncStage {
					point,
					stage: vk::PipelineStageFlags2::ALL_COMMANDS,
				}),
				..QueueWaitOwned::default(arena)
			},
			data: &mut frames[curr_frame],
			buf: vk::CommandBuffer::null(),
			sync: sync.into_iter(),
		}
	}

	pub fn before_pass(&mut self, device: &Device) -> Result<vk::CommandBuffer> {
		let mut sync = self.sync.next().unwrap();
		let buf = self.get_buf(device)?;

		let buf = match (!sync.cross_queue.signal.is_empty(), !sync.cross_queue.wait.is_empty()) {
			// No cross-queue sync.
			(false, false) => {
				emit_barriers(device, buf, &sync.queue.barriers);
				buf
			},
			// Only signal.
			(true, false) => {
				debug_assert!(sync.cross_queue.wait_barriers.is_empty(), "wait barriers without wait");
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				emit_barriers(device, buf, &sync.queue.barriers);

				// The semaphores must be signaled as soon as possible, so submit now.
				self.submit(device, &sync.cross_queue.signal)?
			},
			// Only wait.
			(false, true) => {
				debug_assert!(
					sync.cross_queue.signal_barriers.is_empty(),
					"signal barriers without signal"
				);
				// Everything after this has to wait, so submit the previous work.
				let buf = self.submit(device, &[])?;

				// Make the next buffer wait for whatever is required.
				self.cached_wait.merge(sync.cross_queue.wait);

				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.wait_barriers);
				emit_barriers(device, buf, &sync.queue.barriers);
				buf
			},
			// Both.
			(true, true) => {
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				emit_barriers(device, buf, &sync.queue.barriers);

				// The semaphores must be signaled as soon as possible, so submit now.
				let buf = self.submit(device, &sync.cross_queue.signal)?;

				// Make the next buffer wait for whatever is required.
				// Also emit the post-wait barriers in a separate command because we cannot merge the barriers across
				// submits.
				self.cached_wait.merge(sync.cross_queue.wait);
				emit_barriers(device, buf, &sync.cross_queue.wait_barriers);
				buf
			},
		};

		Ok(buf)
	}

	pub fn finish(mut self, device: &Device) -> Result<()> {
		let mut sync = self.sync.next().unwrap();

		debug_assert!(sync.cross_queue.wait.is_empty(), "Cannot wait after the last pass");

		// Emit all barriers as the last command.
		extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
		emit_barriers(device, self.buf, &sync.queue.barriers);

		// Submit and signal all the semaphores we need to.
		self.data.sync = self.submit_inner(device, &sync.cross_queue.signal)?;

		Ok(())
	}

	fn submit(&mut self, device: &Device, signal: &[SyncStage<vk::Semaphore>]) -> Result<vk::CommandBuffer> {
		if self.buf != vk::CommandBuffer::null() {
			self.submit_inner(device, signal)?;
			self.cached_wait.clear();
			self.buf = vk::CommandBuffer::null();
		}
		self.get_buf(device)
	}

	fn submit_inner(&mut self, device: &Device, signal: &[SyncStage<vk::Semaphore>]) -> Result<SyncPoint<Graphics>> {
		unsafe {
			let span = span!(Level::TRACE, "submit");
			let _e = span.enter();

			device.device().end_command_buffer(self.buf)?;
			device.submit::<Graphics>(self.cached_wait.borrow(), &[self.buf], signal)
		}
	}

	fn get_buf(&mut self, device: &Device) -> Result<vk::CommandBuffer> {
		unsafe {
			if self.buf == vk::CommandBuffer::null() {
				self.buf = self.data.pool.next(device)?;
				device.device().begin_command_buffer(
					self.buf,
					&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)?;
				device.bind_descriptor_set(self.buf, vk::PipelineBindPoint::GRAPHICS);
				device.bind_descriptor_set(self.buf, vk::PipelineBindPoint::COMPUTE);
				device.bind_descriptor_set(self.buf, vk::PipelineBindPoint::RAY_TRACING_KHR);
			}

			Ok(self.buf)
		}
	}
}

fn extend_dep_info(ex: &mut DependencyInfo, other: DependencyInfo) {
	ex.barriers.extend(other.barriers);
	ex.image_barriers.extend(other.image_barriers);
}

fn emit_barriers(device: &Device, buf: vk::CommandBuffer, info: &DependencyInfo) {
	unsafe {
		if !info.is_empty() {
			device.device().cmd_pipeline_barrier2(
				buf,
				&vk::DependencyInfo::default()
					.memory_barriers(&info.barriers)
					.image_memory_barriers(&info.image_barriers),
			);
		}
	}
}
