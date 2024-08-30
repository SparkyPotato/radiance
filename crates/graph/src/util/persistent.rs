use ash::vk;

use crate::{
	device::Device,
	graph::{self, Deletable, ExternalBuffer, ExternalImage},
	resource::{Buffer, BufferDesc, Image, ImageDesc, Resource},
	Result,
};

pub struct PersistentBuffer {
	buffers: [Buffer; 2],
	current: usize,
}

impl PersistentBuffer {
	pub fn new(device: &Device, desc: BufferDesc) -> Result<Self> {
		let buffers = [Buffer::create(device, desc)?, Buffer::create(device, desc)?];
		Ok(Self { buffers, current: 0 })
	}

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

	pub fn desc(&self) -> graph::BufferDesc { self.buffers[0].desc() }

	pub unsafe fn destroy(self, device: &Device) {
		let [r, w] = self.buffers;
		r.destroy(device);
		w.destroy(device);
	}
}

impl Deletable for PersistentBuffer {
	fn into_resources(self, out: &mut Vec<graph::Resource>) {
		let [r, w] = self.buffers;
		r.into_resources(out);
		w.into_resources(out);
	}
}

pub struct PersistentImage {
	images: [Image; 2],
	layout: vk::ImageLayout,
	current: usize,
}

impl PersistentImage {
	pub fn new(device: &Device, desc: ImageDesc) -> Result<Self> {
		let images = [Image::create(device, desc)?, Image::create(device, desc)?];
		Ok(Self {
			images,
			layout: vk::ImageLayout::UNDEFINED,
			current: 0,
		})
	}

	pub fn next(&mut self, final_layout: vk::ImageLayout) -> (ExternalImage, ExternalImage) {
		let next = self.current ^ 1;
		let read = &self.images[self.current];
		let read = ExternalImage {
			handle: read.handle(),
			layout: self.layout,
			desc: read.desc(),
		};
		let write = &self.images[next];
		let write = ExternalImage {
			handle: write.handle(),
			layout: vk::ImageLayout::UNDEFINED,
			desc: write.desc(),
		};
		self.current = next;
		self.layout = final_layout;
		(read, write)
	}

	pub fn desc(&self) -> graph::ImageDesc { self.images[0].desc() }

	pub unsafe fn destroy(self, device: &Device) {
		let [r, w] = self.images;
		r.destroy(device);
		w.destroy(device);
	}
}

impl Deletable for PersistentImage {
	fn into_resources(self, out: &mut Vec<graph::Resource>) {
		let [r, w] = self.images;
		r.into_resources(out);
		w.into_resources(out);
	}
}
