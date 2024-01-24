use radiance_graph::{
	ash::vk,
	graph::{BufferUsage, ExternalBuffer, ReadId, WriteId},
	resource::{BufferDesc, GpuBuffer, GpuBufferHandle, Resource},
	Result,
};

use crate::{CoreBuilder, CoreDevice};

pub struct PersistentBuffer {
	pub buffers: [GpuBuffer; 2],
	pub current: usize,
}

impl PersistentBuffer {
	pub fn new(device: &CoreDevice, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
		let buffers = [
			GpuBuffer::create(device, BufferDesc { size, usage })?,
			GpuBuffer::create(device, BufferDesc { size, usage })?,
		];

		Ok(Self { buffers, current: 0 })
	}

	pub fn size(&self) -> u64 { self.buffers[0].size() }

	pub fn next(
		&mut self, pass: &mut CoreBuilder, read_usage: BufferUsage, write_usage: BufferUsage,
	) -> (ReadId<GpuBufferHandle>, WriteId<GpuBufferHandle>) {
		let next = self.current ^ 1;

		let (read, _) = pass.output(
			ExternalBuffer {
				handle: self.buffers[self.current].handle(),
				prev_usage: None,
				next_usage: None,
			},
			read_usage,
		);
		let (_, write) = pass.output(
			ExternalBuffer {
				handle: self.buffers[next].handle(),
				prev_usage: None,
				next_usage: None,
			},
			write_usage,
		);

		self.current = next;

		(read, write)
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		let [b1, b2] = self.buffers;
		b1.destroy(device);
		b2.destroy(device);
	}
}

