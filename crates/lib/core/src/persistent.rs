use radiance_graph::{
	ash::vk,
	graph::{BufferUsage, ExternalBuffer, ExternalImage, ImageUsage, Res},
	resource::{BufferDesc, GpuBuffer, GpuBufferHandle, Image, ImageDesc, ImageView, Resource},
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
	) -> (Res<GpuBufferHandle>, Res<GpuBufferHandle>) {
		let next = self.current ^ 1;

		let read = pass.output(
			ExternalBuffer {
				handle: self.buffers[self.current].handle(),
				wait: None,
				signal: None,
			},
			read_usage,
		);
		let write = pass.output(
			ExternalBuffer {
				handle: self.buffers[next].handle(),
				wait: None,
				signal: None,
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

#[derive(Default)]
pub struct PersistentImage {
	pub images: [Image; 2],
	pub current: usize,
	pub desc: ImageDesc,
}

impl PersistentImage {
	pub fn new(device: &CoreDevice, desc: ImageDesc) -> Result<Self> {
		let images = [Image::create(device, desc)?, Image::create(device, desc)?];

		Ok(Self {
			images,
			current: 0,
			desc,
		})
	}

	pub fn next(
		&mut self, pass: &mut CoreBuilder, read_usage: ImageUsage, write_usage: ImageUsage,
	) -> (Res<ImageView>, Res<ImageView>) {
		let next = self.current ^ 1;

		let read = pass.output(
			ExternalImage {
				handle: self.images[self.current].handle(),
				size: self.desc.size,
				levels: self.desc.levels,
				layers: self.desc.layers,
				samples: self.desc.samples,
				wait: None,
				signal: None,
			},
			read_usage,
		);
		let write = pass.output(
			ExternalImage {
				handle: self.images[next].handle(),
				size: self.desc.size,
				levels: self.desc.levels,
				layers: self.desc.layers,
				samples: self.desc.samples,
				wait: None,
				signal: None,
			},
			write_usage,
		);

		self.current = next;

		(read, write)
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		let [b1, b2] = self.images;
		b1.destroy(device);
		b2.destroy(device);
	}
}
