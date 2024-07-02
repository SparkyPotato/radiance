use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	cmd::CommandPool,
	device::{Device, Graphics, QueueWaitOwned, SyncPoint, SyncStage},
	graph::compile::{DependencyInfo, EventInfo, QueueSync, Sync},
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
			pool: CommandPool::new(device, device.queue_families().graphics)?,
		})
	}

	pub fn reset(&mut self, device: &Device) -> Result<()> {
		unsafe {
			// Let GPU finish this frame before doing anything else.
			self.sync.wait(device)?;
			self.pool.reset(device)?;

			Ok(())
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		unsafe {
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

	pub fn pass(&mut self, device: &Device) -> Result<vk::CommandBuffer> {
		let mut sync = self.sync.next().unwrap();

		match (!sync.cross_queue.signal.is_empty(), !sync.cross_queue.wait.is_empty()) {
			// No cross-queue sync.
			(false, false) => {
				self.start_buf(device)?; // May be the first pass, ensure the buffer is started.
				emit_queue_sync(
					device,
					self.cached_wait.binary_semaphores.allocator(),
					self.buf,
					&sync.queue,
				);
			},
			// Only signal.
			(true, false) => {
				debug_assert!(
					self.buf != vk::CommandBuffer::null(),
					"Cannot signal before the first pass"
				);

				// Emit the pre-signal barriers, but also the main queue barriers so we can save on a
				// `vkCmdPipelineBarrier`.
				// If there were any wait barriers, we can also emit them since the non-existent wait is
				// already done.
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.wait_barriers);
				emit_queue_sync(
					device,
					self.cached_wait.binary_semaphores.allocator(),
					self.buf,
					&sync.queue,
				);

				// The semaphores must be signaled as soon as possible, so submit now.
				self.submit(device, &sync.cross_queue.signal)?;
			},
			// Only wait.
			(false, true) => {
				if self.buf != vk::CommandBuffer::null() {
					// If there was a previous pass, submit it now.
					self.submit(device, &[])?;
				} else {
					self.start_buf(device)?;
				}

				// Make the next buffer wait for whatever is required.
				self.cached_wait.merge(sync.cross_queue.wait);

				// Emit the post-wait barriers, but also the main queue barriers so we can save on a
				// `vkCmdPipelineBarrier`.
				// If there were any signal barriers, we can also emit them since the non-existent signal will happen
				// soon.
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.wait_barriers);
				emit_queue_sync(
					device,
					self.cached_wait.binary_semaphores.allocator(),
					self.buf,
					&sync.queue,
				);
			},
			// Both.
			(true, true) => {
				debug_assert!(
					self.buf != vk::CommandBuffer::null(),
					"Cannot signal before the first pass"
				);

				// Emit the pre-signal barriers, but also the main queue barriers so we can save on a
				// `vkCmdPipelineBarrier`.
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				emit_queue_sync(
					device,
					self.cached_wait.binary_semaphores.allocator(),
					self.buf,
					&sync.queue,
				);

				// The semaphores must be signaled as soon as possible, so submit now.
				self.submit(device, &sync.cross_queue.signal)?;

				// Make the next buffer wait for whatever is required.
				// Also emit the post-wait barriers in a separate command because we cannot merge the barriers across
				// submits.
				self.cached_wait.merge(sync.cross_queue.wait);
				emit_barriers(device, self.buf, &sync.cross_queue.wait_barriers);
			},
		}

		Ok(self.buf)
	}

	pub fn finish(mut self, device: &Device, pre_submit: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
		let mut sync = self.sync.next().unwrap();

		debug_assert!(sync.cross_queue.wait.is_empty(), "Cannot wait after the last pass");

		// Emit all barriers as the last command.
		extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
		emit_queue_sync(
			device,
			self.cached_wait.binary_semaphores.allocator(),
			self.buf,
			&sync.queue,
		);

		// Submit and signal all the semaphores we need to.
		pre_submit(self.buf);
		self.data.sync = self.submit_inner(device, &sync.cross_queue.signal)?;

		Ok(())
	}

	fn submit(&mut self, device: &Device, signal: &[SyncStage<vk::Semaphore>]) -> Result<()> {
		self.submit_inner(device, signal)?;
		self.cached_wait.clear();
		self.buf = vk::CommandBuffer::null();
		self.start_buf(device)?;

		Ok(())
	}

	fn submit_inner(&mut self, device: &Device, signal: &[SyncStage<vk::Semaphore>]) -> Result<SyncPoint<Graphics>> {
		unsafe {
			let span = span!(Level::TRACE, "submit");
			let _e = span.enter();

			device.device().end_command_buffer(self.buf)?;
			device.submit::<Graphics>(self.cached_wait.borrow(), &[self.buf], signal, vk::Fence::null())
		}
	}

	fn start_buf(&mut self, device: &Device) -> Result<()> {
		unsafe {
			if self.buf == vk::CommandBuffer::null() {
				self.buf = self.data.pool.next(device)?;
				device.device().begin_command_buffer(
					self.buf,
					&vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)?;
			}

			Ok(())
		}
	}
}

fn extend_dep_info(ex: &mut DependencyInfo, other: DependencyInfo) {
	ex.barriers.extend(other.barriers);
	ex.image_barriers.extend(other.image_barriers);
}

fn dependency_info<'a>(info: &'a DependencyInfo) -> vk::DependencyInfoBuilder<'a> {
	vk::DependencyInfo::builder()
		.memory_barriers(&info.barriers)
		.image_memory_barriers(&info.image_barriers)
}

fn emit_queue_sync(device: &Device, arena: &Arena, buf: vk::CommandBuffer, sync: &QueueSync) {
	emit_barriers(device, buf, &sync.barriers);
	emit_event_waits(device, arena, buf, &sync.wait_events);
	emit_event_sets(device, buf, &sync.set_events);
}

fn emit_barriers(device: &Device, buf: vk::CommandBuffer, info: &DependencyInfo) {
	unsafe {
		if !info.barriers.is_empty() || !info.image_barriers.is_empty() {
			device.device().cmd_pipeline_barrier2(buf, &dependency_info(info));
		}
	}
}

fn emit_event_sets(device: &Device, buf: vk::CommandBuffer, events: &[EventInfo]) {
	for event in events {
		unsafe {
			device
				.device()
				.cmd_set_event2(buf, event.event, &dependency_info(&event.info));
		}
	}
}

fn emit_event_waits(device: &Device, arena: &Arena, buf: vk::CommandBuffer, events: &[EventInfo]) {
	if !events.is_empty() {
		let infos: Vec<_, _> = events
			.iter()
			.map(|e| dependency_info(&e.info).build())
			.collect_in(arena);
		let events: Vec<_, _> = events.iter().map(|e| e.event).collect_in(arena);

		unsafe {
			device.device().cmd_wait_events2(buf, &events, &infos);
		}
	}
}
