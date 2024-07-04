use radiance_graph::{ash::vk, device::Device};

pub fn semaphore(device: &Device) -> vk::Semaphore {
	unsafe {
		device
			.device()
			.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
			.unwrap()
	}
}
