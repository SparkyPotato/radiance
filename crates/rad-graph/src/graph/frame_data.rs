use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::Arena,
	cmd::CommandPool,
	device::{Device, Graphics, QueueWaitOwned, SyncPoint, SyncStage},
	graph::compile::{DependencyInfo, QueueSync, Sync},
	resource::{Buffer, Image, ImageView, Resource as _, AS},
	Result,
};

pub trait Deletable {
	fn into_resources(self, out: &mut Vec<Resource>);
}

pub enum Resource {
	Buffer(Buffer),
	Image(Image),
	ImageView(ImageView),
	AS(AS),
}

impl Resource {
	pub unsafe fn destroy(self, device: &Device) {
		match self {
			Resource::Buffer(x) => x.destroy(device),
			Resource::Image(x) => x.destroy(device),
			Resource::ImageView(x) => x.destroy(device),
			Resource::AS(x) => x.destroy(device),
		}
	}
}

pub struct FrameData {
	sync: SyncPoint<Graphics>,
	pool: CommandPool,
	delete_queue: Vec<Resource>,
}

impl FrameData {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			sync: SyncPoint::default(),
			pool: CommandPool::new(device, device.queue_families().into::<Graphics>())?,
			delete_queue: Vec::new(),
		})
	}

	pub fn delete(&mut self, res: impl Deletable) { res.into_resources(&mut self.delete_queue); }

	pub fn reset(&mut self, device: &Device) -> Result<()> {
		unsafe {
			let span = span!(Level::TRACE, "wait for gpu");
			let _e = span.enter();
			// Let GPU finish this frame before doing anything else.
			self.sync.wait(device)?;
			self.pool.reset(device)?;
			for r in self.delete_queue.drain(..) {
				r.destroy(device);
			}

			Ok(())
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		unsafe {
			// Let GPU finish this frame before doing anything else.
			let _ = self.sync.wait(device);
			self.pool.destroy(device);
			for r in self.delete_queue {
				r.destroy(device);
			}
		}
	}
}

impl Deletable for Buffer {
	fn into_resources(self, out: &mut Vec<Resource>) { out.push(Resource::Buffer(self)); }
}

impl Deletable for Image {
	fn into_resources(self, out: &mut Vec<Resource>) { out.push(Resource::Image(self)); }
}

impl Deletable for ImageView {
	fn into_resources(self, out: &mut Vec<Resource>) { out.push(Resource::ImageView(self)); }
}

impl Deletable for AS {
	fn into_resources(self, out: &mut Vec<Resource>) { out.push(Resource::AS(self)); }
}

impl Deletable for Resource {
	fn into_resources(self, out: &mut Vec<Resource>) { out.push(self) }
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
				emit_queue_sync(device, self.buf, &sync.queue);
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
				emit_queue_sync(device, self.buf, &sync.queue);

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
				emit_queue_sync(device, self.buf, &sync.queue);
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
				emit_queue_sync(device, self.buf, &sync.queue);

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

	pub fn finish(mut self, device: &Device) -> Result<()> {
		let mut sync = self.sync.next().unwrap();

		debug_assert!(sync.cross_queue.wait.is_empty(), "Cannot wait after the last pass");

		// Emit all barriers as the last command.
		extend_dep_info(&mut sync.queue.barriers, sync.cross_queue.signal_barriers);
		emit_queue_sync(device, self.buf, &sync.queue);

		// Submit and signal all the semaphores we need to.
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
				let dev = device.device();
				dev.begin_command_buffer(
					self.buf,
					&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)?;
				dev.cmd_bind_descriptor_sets(
					self.buf,
					vk::PipelineBindPoint::GRAPHICS,
					device.layout(),
					0,
					&[device.descriptor_set()],
					&[],
				);
				dev.cmd_bind_descriptor_sets(
					self.buf,
					vk::PipelineBindPoint::COMPUTE,
					device.layout(),
					0,
					&[device.descriptor_set()],
					&[],
				);
				dev.cmd_bind_descriptor_sets(
					self.buf,
					vk::PipelineBindPoint::RAY_TRACING_KHR,
					device.layout(),
					0,
					&[device.descriptor_set()],
					&[],
				);
			}

			Ok(())
		}
	}
}

fn extend_dep_info(ex: &mut DependencyInfo, other: DependencyInfo) {
	ex.barriers.extend(other.barriers);
	ex.image_barriers.extend(other.image_barriers);
}

fn dependency_info<'a>(info: &'a DependencyInfo) -> vk::DependencyInfo<'a> {
	vk::DependencyInfo::default()
		.memory_barriers(&info.barriers)
		.image_memory_barriers(&info.image_barriers)
}

fn emit_queue_sync(device: &Device, buf: vk::CommandBuffer, sync: &QueueSync) {
	emit_barriers(device, buf, &sync.barriers);
}

fn emit_barriers(device: &Device, buf: vk::CommandBuffer, info: &DependencyInfo) {
	unsafe {
		if !info.barriers.is_empty() || !info.image_barriers.is_empty() {
			device.device().cmd_pipeline_barrier2(buf, &dependency_info(info));
		}
	}
}
