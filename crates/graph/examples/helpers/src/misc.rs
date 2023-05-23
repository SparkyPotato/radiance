use radiance_graph::{
	ash::vk::{Extent2D, Rect2D, Semaphore, SemaphoreCreateInfo},
	device::Device,
};
use vek::Vec2;

pub fn simple_rect(size: Vec2<u32>) -> Rect2D {
	Rect2D::builder()
		.extent(Extent2D::builder().width(size.x).height(size.y).build())
		.build()
}

pub fn semaphore(device: &Device) -> Semaphore {
	unsafe {
		device
			.device()
			.create_semaphore(&SemaphoreCreateInfo::builder().build(), None)
			.unwrap()
	}
}
