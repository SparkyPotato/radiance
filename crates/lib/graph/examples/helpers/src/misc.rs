use radiance_graph::{ash::vk, device::Device};
use vek::Vec2;

pub fn simple_rect(size: Vec2<u32>) -> vk::Rect2D {
	vk::Rect2D::builder()
		.extent(vk::Extent2D::builder().width(size.x).height(size.y).build())
		.build()
}

pub fn semaphore(device: &Device) -> vk::Semaphore {
	unsafe {
		device
			.device()
			.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
			.unwrap()
	}
}
