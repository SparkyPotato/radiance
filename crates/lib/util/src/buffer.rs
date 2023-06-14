use ash::vk;
use radiance_graph::{
	device::{Device, QueueType},
	resource::{BufferDesc, GpuBuffer, Resource},
	Result,
};

use crate::{deletion::DeletionQueue, staging::StagingCtx};

/// A GPU-only buffer that works like a `Vec`.
pub struct StretchyBuffer {
	inner: GpuBuffer,
	usage: vk::BufferUsageFlags,
	len: u64,
}

impl StretchyBuffer {
	pub fn new(device: &Device, desc: BufferDesc) -> Result<Self> {
		let inner = GpuBuffer::create(
			device,
			BufferDesc {
				usage: vk::BufferUsageFlags::TRANSFER_DST | desc.usage,
				..desc
			},
		)?;
		Ok(Self {
			inner,
			usage: desc.usage,
			len: 0,
		})
	}

	pub fn push(&mut self, ctx: &mut StagingCtx, queue: &mut DeletionQueue, data: &[u8]) -> Result<u64> {
		let len = data.len() as u64;
		self.reserve(ctx, queue, self.len + len)?;
		let offset = self.len;
		ctx.stage_buffer(data, self.inner.inner.inner(), offset as _)?;
		self.len += len;
		Ok(offset)
	}

	pub fn reserve(&mut self, ctx: &mut StagingCtx, queue: &mut DeletionQueue, bytes: u64) -> Result<()> {
		if self.inner.inner.size() < bytes {
			let new_size = bytes * 2;
			let new_buffer = GpuBuffer::create(
				ctx.device,
				BufferDesc {
					size: new_size as _,
					usage: vk::BufferUsageFlags::TRANSFER_DST | self.usage,
				},
			)?;
			unsafe {
				let buf = ctx.execute_before(QueueType::Transfer)?;
				ctx.device.device().cmd_copy_buffer(
					buf,
					self.inner.inner.inner(),
					new_buffer.inner.inner(),
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size: self.len as _,
					}],
				);
				let old = std::mem::replace(&mut self.inner, new_buffer);
				queue.delete(old);
			}
		}

		Ok(())
	}

	pub fn inner(&self) -> &GpuBuffer { &self.inner }

	pub fn len(&self) -> u64 { self.len }

	pub unsafe fn destroy(self, queue: &mut DeletionQueue) { queue.delete(self.inner); }
}
