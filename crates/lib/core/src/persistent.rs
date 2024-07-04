use radiance_graph::{
	ash::vk,
	graph::{self, BufferUsage, ExternalBuffer, ExternalImage, ImageUsage, Res},
	resource::{Buffer, BufferDesc, GpuBufferHandle, Image, ImageDesc, ImageView, Resource},
	Result,
};

use crate::{CoreBuilder, CoreDevice};

pub struct PersistentBuffer {
	pub buffers: [GpuBuffer; 2],
	pub current: usize,
}

impl PersistentBuffer {
	pub fn new(device: &CoreDevice, desc: BufferDesc) -> Result<Self> {
		let buffers = [Buffer::create(device, desc)?, Buffer::create(device, desc)?];

		Ok(Self { buffers, current: 0 })
	}

	pub fn size(&self) -> u64 { self.buffers[0].size() }

	pub fn next(&mut self) -> (ExternalBuffer, ExternalBuffer) {
		let next = self.current ^ 1;

		let read = ExternalBuffer {
			handle: self.buffers[self.current].handle(),
		};
		let write = ExternalBuffer {
			handle: self.buffers[next].handle(),
		};

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
	pub desc: graph::ImageDesc,
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

	pub fn next(&mut self) -> (ExternalImage, ExternalImage) {
		let next = self.current ^ 1;

		let read = ExternalImage {
			handle: self.images[self.current].handle(),
			desc: self.desc,
			format: self.desc.format,
		};
		let write = ExternalImage {
			handle: self.images[next].handle(),
		};

		self.current = next;

		(read, write)
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		let [b1, b2] = self.images;
		b1.destroy(device);
		b2.destroy(device);
	}
}
