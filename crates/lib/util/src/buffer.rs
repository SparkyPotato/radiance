use std::ops::Deref;

use ash::vk;
use radiance_graph::{
	device::{Device, QueueType},
	resource::{BufferDesc, GpuBuffer, Resource},
	Result,
};
use range_alloc::RangeAllocator;

use crate::{
	deletion::{DeletionQueue, IntoResource},
	staging::StagingCtx,
};

pub struct BufSpan {
	pub offset: u64,
	pub size: u64,
}

pub struct AllocBuffer {
	inner: GpuBuffer,
	usage: vk::BufferUsageFlags,
	alloc: RangeAllocator<u64>,
}

impl Deref for AllocBuffer {
	type Target = GpuBuffer;

	fn deref(&self) -> &Self::Target { &self.inner }
}

impl AllocBuffer {
	pub fn new(device: &Device, desc: BufferDesc) -> Result<Self> {
		let inner = GpuBuffer::create(
			device,
			BufferDesc {
				usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | desc.usage,
				..desc
			},
		)?;

		Ok(Self {
			inner,
			usage: desc.usage,
			alloc: RangeAllocator::new(0..desc.size),
		})
	}

	pub fn alloc(&mut self, ctx: &mut StagingCtx, queue: &mut DeletionQueue, data: &[u8]) -> Result<BufSpan> {
		let len = data.len() as u64;
		let span = self.alloc_size(ctx, queue, len)?;
		unsafe {
			std::ptr::copy_nonoverlapping(
				data.as_ptr(),
				self.data().as_ptr().cast::<u8>().add(span.offset as usize),
				data.len(),
			);
		}

		Ok(span)
	}

	pub fn alloc_size(&mut self, ctx: &mut StagingCtx, queue: &mut DeletionQueue, size: u64) -> Result<BufSpan> {
		match self.alloc.allocate_range(size) {
			Ok(range) => Ok(BufSpan {
				offset: range.start,
				size,
			}),
			Err(_) => {
				self.reserve(ctx, queue, self.inner.size() + size)?;
				self.alloc_size(ctx, queue, size)
			},
		}
	}

	pub fn dealloc(&mut self, span: BufSpan) { self.alloc.free_range(span.offset..span.offset + span.size); }

	pub fn reserve(&mut self, ctx: &mut StagingCtx, queue: &mut DeletionQueue, bytes: u64) -> Result<()> {
		if self.inner.size() < bytes {
			let mut new_size = self.inner.size();
			while new_size < bytes {
				new_size *= 2;
			}
			let new_buffer = GpuBuffer::create(
				ctx.device,
				BufferDesc {
					size: new_size as _,
					usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | self.usage,
				},
			)?;
			unsafe {
				let buf = ctx.execute_before(QueueType::Transfer)?;
				for r in self.alloc.allocated_ranges() {
					ctx.device.device().cmd_copy_buffer(
						buf,
						self.inner.inner(),
						new_buffer.inner(),
						&[vk::BufferCopy {
							src_offset: r.start,
							dst_offset: r.start,
							size: r.end - r.start,
						}],
					)
				}
				self.alloc.grow_to(new_size);
				let old = std::mem::replace(&mut self.inner, new_buffer);
				queue.delete(old);
			}
		}

		Ok(())
	}

	pub fn clear(&mut self) { self.alloc.reset(); }

	pub unsafe fn delete(self, queue: &mut DeletionQueue) { queue.delete(self.inner); }

	pub unsafe fn destroy(self, device: &Device) { self.inner.destroy(device); }
}

impl IntoResource for AllocBuffer {
	fn into_resource(self) -> crate::deletion::Resource { self.inner.into_resource() }
}

