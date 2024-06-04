use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	cmd::CommandPool,
	device::{Device, GraphicsSyncPoint},
	graph::compile::{DependencyInfo, EventInfo, QueueSync, Sync},
	Result,
};

pub struct FrameData {
	pool: CommandPool,
	sync: GraphicsSyncPoint,
}

impl FrameData {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pool: CommandPool::new(device, device.queue_families().graphics)?,
			sync: GraphicsSyncPoint::none(),
		})
	}

	pub fn reset(&mut self, device: &Device) -> Result<()> {
		unsafe {
			// Let GPU finish this frame before doing anything else.
			device.wait(self.sync)?;
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
	data: &'a mut FrameData,
	buf: vk::CommandBuffer,
	sync: I,
	cached_wait: Vec<vk::SemaphoreSubmitInfo, &'a Arena>,
	first: bool,
}

impl<'a, I: Iterator<Item = Sync<'a>>> Submitter<'a, I> {
	pub fn new(
		arena: &'a Arena, sync: impl IntoIterator<IntoIter = I>, frames: &'a mut [FrameData], curr_frame: usize,
	) -> Self {
		Self {
			cached_wait: Vec::new_in(arena),
			data: &mut frames[curr_frame],
			buf: vk::CommandBuffer::null(),
			sync: sync.into_iter(),
			first: true,
		}
	}

	pub fn pass(&mut self, device: &Device) -> Result<vk::CommandBuffer> {
		let mut sync = self.sync.next().unwrap();

		match (
			!sync.cross_queue.signal_semaphores.is_empty(),
			!sync.cross_queue.wait_semaphores.is_empty(),
		) {
			// No cross-queue sync.
			(false, false) => {
				self.start_buf(device)?; // May be the first pass, ensure the buffer is started.
				emit_queue_sync(device, self.cached_wait.allocator(), self.buf, &sync.queue);
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
				emit_queue_sync(device, self.cached_wait.allocator(), self.buf, &sync.queue);

				// The semaphores must be signaled as soon as possible, so submit now.
				self.submit(device, &sync.cross_queue.signal_semaphores)?;
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
				self.cached_wait.extend(sync.cross_queue.wait_semaphores);

				// Emit the post-wait barriers, but also the main queue barriers so we can save on a
				// `vkCmdPipelineBarrier`.
				// If there were any signal barriers, we can also emit them since the non-existent signal will happen
				// soon.
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
				extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.wait_barriers);
				emit_queue_sync(device, self.cached_wait.allocator(), self.buf, &sync.queue);
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
				emit_queue_sync(device, self.cached_wait.allocator(), self.buf, &sync.queue);

				// The semaphores must be signaled as soon as possible, so submit now.
				self.submit(device, &sync.cross_queue.signal_semaphores)?;

				// Make the next buffer wait for whatever is required.
				// Also emit the post-wait barriers in a separate command because we cannot merge the barriers across
				// submits.
				self.cached_wait.extend(sync.cross_queue.wait_semaphores);
				emit_barriers(device, self.buf, &sync.cross_queue.wait_barriers);
			},
		}

		Ok(self.buf)
	}

	pub fn finish(mut self, device: &Device, pre_submit: impl FnOnce(vk::CommandBuffer)) -> Result<GraphicsSyncPoint> {
		let mut sync = self.sync.next().unwrap();
		debug_assert!(
			sync.cross_queue.wait_semaphores.is_empty(),
			"Cannot wait after the last pass"
		);

		// Emit all barriers as the last command.
		extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
		emit_queue_sync(device, self.cached_wait.allocator(), self.buf, &sync.queue);

		// Submit and signal all the semaphores we need to.
		pre_submit(self.buf);
		self.submit_inner(device, &sync.cross_queue.signal_semaphores, true)
			.map(|s| s.unwrap())
	}

	fn submit(&mut self, device: &Device, signal: &[vk::SemaphoreSubmitInfo]) -> Result<()> {
		self.submit_inner(device, signal, false)?;
		self.cached_wait.clear();
		self.buf = vk::CommandBuffer::null();
		self.start_buf(device)?;

		Ok(())
	}

	fn submit_inner(
		&mut self, device: &Device, signal: &[vk::SemaphoreSubmitInfo], sync: bool,
	) -> Result<Option<GraphicsSyncPoint>> {
		unsafe {
			let span = span!(Level::TRACE, "submit");
			let _e = span.enter();

			device.device().end_command_buffer(self.buf)?;

			let cmd = &[vk::CommandBufferSubmitInfo::builder().command_buffer(self.buf).build()];
			let i = &[vk::SubmitInfo2::builder()
				.wait_semaphore_infos(&self.cached_wait)
				.command_buffer_infos(cmd)
				.signal_semaphore_infos(signal)
				.build()];
			let b = device.submit_graphics(i);
			let b = if self.first {
				self.first = false;
				b.wait_on_prev(vk::PipelineStageFlags2::ALL_COMMANDS)
			} else {
				b
			};
			if sync {
				b.submit_signal(vk::PipelineStageFlags2::ALL_COMMANDS).map(Some)
			} else {
				b.submit().map(|_| None)
			}
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
