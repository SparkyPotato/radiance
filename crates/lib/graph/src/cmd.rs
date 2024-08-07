use ash::vk;

use crate::{device::Device, Result};

pub struct CommandPool {
	pool: vk::CommandPool,
	bufs: Vec<vk::CommandBuffer>,
	buf_cursor: usize,
}

impl CommandPool {
	pub fn new(device: &Device, queue: u32) -> Result<Self> {
		let pool = unsafe {
			device.device().create_command_pool(
				&vk::CommandPoolCreateInfo::default()
					.queue_family_index(queue)
					.flags(vk::CommandPoolCreateFlags::TRANSIENT),
				None,
			)
		}?;

		Ok(Self {
			pool,
			bufs: Vec::new(),
			buf_cursor: 0,
		})
	}

	/// Reset the command pool.
	///
	/// # Safety
	/// All command buffers previously allocated by a call to [`Self::next`] must not be used anymore.
	pub unsafe fn reset(&mut self, device: &Device) -> Result<()> {
		device
			.device()
			.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
		self.buf_cursor = 0; // We can now hand out the first buffer again.

		Ok(())
	}

	pub fn next(&mut self, device: &Device) -> Result<vk::CommandBuffer> {
		if let Some(buf) = self.bufs.get(self.buf_cursor) {
			self.buf_cursor += 1;
			Ok(*buf)
		} else {
			let buf = unsafe {
				device.device().allocate_command_buffers(
					&vk::CommandBufferAllocateInfo::default()
						.command_pool(self.pool)
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_buffer_count(1),
				)
			}?[0];
			self.buf_cursor += 1;
			self.bufs.push(buf);
			Ok(buf)
		}
	}

	/// # Safety
	/// This invalidates all command buffers allocated from this pool, and must also be synchronized correctly.
	pub unsafe fn destroy(self, device: &Device) { device.device().destroy_command_pool(self.pool, None); }
}
