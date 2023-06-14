use radiance_graph::{
	device::Device,
	graph::FRAMES_IN_FLIGHT,
	resource::{Buffer, GpuBuffer, Image, ImageView, Resource as R, UploadBuffer},
};

pub enum Resource {
	Buffer(Buffer),
	Image(Image),
	ImageView(ImageView),
}

pub trait IntoResource {
	fn into_resource(self) -> Resource;
}

impl IntoResource for UploadBuffer {
	fn into_resource(self) -> Resource { Resource::Buffer(self.inner) }
}

impl IntoResource for GpuBuffer {
	fn into_resource(self) -> Resource { Resource::Buffer(self.inner) }
}

impl IntoResource for Image {
	fn into_resource(self) -> Resource { Resource::Image(self) }
}

impl IntoResource for ImageView {
	fn into_resource(self) -> Resource { Resource::ImageView(self) }
}

pub struct DeletionQueue {
	queues: [Vec<Resource>; FRAMES_IN_FLIGHT + 1],
	curr: usize,
}

impl DeletionQueue {
	pub fn new() -> Self {
		Self {
			queues: Default::default(),
			curr: 0,
		}
	}

	/// # Safety
	/// The resource must not be used after this function is called. It must also be synchronized with calls to
	/// `advance`, as those will destroy the resource.
	pub unsafe fn delete(&mut self, resource: impl IntoResource) {
		self.queues[self.curr].push(resource.into_resource());
	}

	/// # Safety
	/// All work on the GPU must be done at this point.
	pub unsafe fn destroy(self, device: &Device) {
		for queue in self.queues {
			for x in queue {
				match x {
					Resource::Buffer(x) => x.destroy(device),
					Resource::Image(x) => x.destroy(device),
					Resource::ImageView(x) => x.destroy(device),
				}
			}
		}
	}

	pub fn advance(&mut self, device: &Device) {
		self.curr = (self.curr + 1) % self.queues.len();
		for x in self.queues[self.curr].drain(..) {
			match x {
				Resource::Buffer(x) => unsafe { x.destroy(device) },
				Resource::Image(x) => unsafe { x.destroy(device) },
				Resource::ImageView(x) => unsafe { x.destroy(device) },
			}
		}
	}
}
