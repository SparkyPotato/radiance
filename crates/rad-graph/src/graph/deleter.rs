use crate::{
	device::Device,
	graph::FRAMES_IN_FLIGHT,
	resource::{Buffer, Image, ImageView, Resource as _, AS},
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
	pub unsafe fn destroy(self, device: &Device) { unsafe {
		match self {
			Resource::Buffer(x) => x.destroy(device),
			Resource::Image(x) => x.destroy(device),
			Resource::ImageView(x) => x.destroy(device),
			Resource::AS(x) => x.destroy(device),
		}
	}}
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

pub struct Deleter {
	queues: [Vec<Resource>; FRAMES_IN_FLIGHT + 1],
	curr: usize,
}

impl Default for Deleter {
    fn default() -> Self {
        Self::new()
    }
}

impl Deleter {
	pub fn new() -> Self {
		Self {
			queues: Default::default(),
			curr: 0,
		}
	}

	pub fn push(&mut self, item: impl Deletable) { item.into_resources(&mut self.queues[self.curr]); }

	pub fn next(&mut self, device: &Device) {
		self.curr = (self.curr + 1) % self.queues.len();
		for resource in self.queues[self.curr].drain(..) {
			unsafe { resource.destroy(device) };
		}
	}

	pub unsafe fn destroy(self, device: &Device) { unsafe {
		for queue in self.queues {
			for resource in queue {
				resource.destroy(device);
			}
		}
	}}
}
