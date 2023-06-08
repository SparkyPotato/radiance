use radiance_graph::{
	ash::{extensions::khr, vk},
	device::Device,
	graph::{ExecutionSnapshot, ExternalImage, ExternalSync, ImageUsageType, RenderGraph},
};
use winit::window::Window;

use crate::misc::semaphore;

struct OldSwapchain {
	swapchain: vk::SwapchainKHR,
	snapshot: Option<ExecutionSnapshot>,
}

pub struct Swapchain {
	surface: vk::SurfaceKHR,
	swapchain_ext: khr::Swapchain,
	swapchain: vk::SwapchainKHR,
	old_swapchain: OldSwapchain,
	images: Vec<vk::Image>,
	available: vk::Semaphore,
	rendered: vk::Semaphore,
}

impl Swapchain {
	pub fn new(device: &Device, surface: vk::SurfaceKHR, window: &Window) -> Self {
		let swapchain_ext = khr::Swapchain::new(device.instance(), device.device());
		let mut this = Self {
			surface,
			swapchain_ext,
			swapchain: vk::SwapchainKHR::null(),
			old_swapchain: OldSwapchain {
				swapchain: vk::SwapchainKHR::null(),
				snapshot: None,
			},
			images: Vec::new(),
			available: semaphore(device),
			rendered: semaphore(device),
		};
		this.make(window);
		this
	}

	pub fn acquire(&self) -> (ExternalImage, u32) {
		unsafe {
			let (id, _) = self
				.swapchain_ext
				.acquire_next_image(self.swapchain, u64::MAX, self.available, vk::Fence::null())
				.unwrap();
			(
				ExternalImage {
					handle: self.images[id as usize],
					prev_usage: Some(ExternalSync {
						semaphore: self.available,
						usage: &[ImageUsageType::Present],
						..Default::default()
					}),
					next_usage: Some(ExternalSync {
						semaphore: self.rendered,
						usage: &[ImageUsageType::Present],
						..Default::default()
					}),
				},
				id,
			)
		}
	}

	pub fn present(&mut self, device: &Device, id: u32, graph: &RenderGraph) {
		unsafe {
			if let Some(snapshot) = self.old_swapchain.snapshot.take() {
				if snapshot.is_complete(graph) {
					self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
					self.old_swapchain = OldSwapchain {
						swapchain: vk::SwapchainKHR::null(),
						snapshot: None,
					};
				}
			}

			self.swapchain_ext
				.queue_present(
					*device.graphics_queue(),
					&vk::PresentInfoKHR::builder()
						.wait_semaphores(&[self.rendered])
						.swapchains(&[self.swapchain])
						.image_indices(&[id])
						.build(),
				)
				.unwrap();
		}
	}

	pub fn resize(&mut self, device: &Device, window: &Window, graph: &RenderGraph) {
		if let Some(snapshot) = self.old_swapchain.snapshot.take() {
			snapshot.wait(device).unwrap();
			unsafe {
				self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			}
		}

		self.old_swapchain = OldSwapchain {
			swapchain: self.swapchain,
			snapshot: Some(graph.snapshot()),
		};
		self.make(window);
	}

	fn make(&mut self, window: &Window) {
		unsafe {
			self.swapchain = self
				.swapchain_ext
				.create_swapchain(
					&vk::SwapchainCreateInfoKHR::builder()
						.surface(self.surface)
						.min_image_count(2)
						.image_format(vk::Format::B8G8R8A8_SRGB)
						.image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
						.image_extent(vk::Extent2D {
							width: window.inner_size().width,
							height: window.inner_size().height,
						})
						.image_array_layers(1)
						.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST) // Check
						.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
						.pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
						.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
						.present_mode(vk::PresentModeKHR::FIFO)
						.old_swapchain(self.old_swapchain.swapchain)
						.clipped(true),
					None,
				)
				.unwrap();

			self.images = self.swapchain_ext.get_swapchain_images(self.swapchain).unwrap();
		}
	}

	pub fn destroy(&mut self, device: &Device) {
		unsafe {
			self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			self.swapchain_ext.destroy_swapchain(self.swapchain, None);
			device.device().destroy_semaphore(self.available, None);
			device.device().destroy_semaphore(self.rendered, None);
		}
	}
}
