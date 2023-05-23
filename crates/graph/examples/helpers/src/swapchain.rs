use radiance_graph::{
	ash::{
		extensions::khr,
		vk::{
			ColorSpaceKHR,
			CompositeAlphaFlagsKHR,
			Extent2D,
			Fence,
			Format,
			Image,
			ImageUsageFlags,
			PresentInfoKHR,
			PresentModeKHR,
			Semaphore,
			SharingMode,
			SurfaceKHR,
			SurfaceTransformFlagsKHR,
			SwapchainCreateInfoKHR,
			SwapchainKHR,
		},
	},
	device::Device,
	graph::{ExecutionSnapshot, ExternalImage, ExternalSync, ImageUsageType, RenderGraph},
};
use winit::window::Window;

use crate::misc::semaphore;

struct OldSwapchain {
	swapchain: SwapchainKHR,
	snapshot: Option<ExecutionSnapshot>,
}

pub struct Swapchain {
	swapchain_ext: khr::Swapchain,
	swapchain: SwapchainKHR,
	old_swapchain: OldSwapchain,
	images: Vec<Image>,
	available: Semaphore,
	rendered: Semaphore,
}

impl Swapchain {
	pub fn new(device: &Device, surface: SurfaceKHR, window: &Window) -> Self {
		let swapchain_ext = khr::Swapchain::new(device.instance(), device.device());
		let mut this = Self {
			swapchain_ext,
			swapchain: SwapchainKHR::null(),
			old_swapchain: OldSwapchain {
				swapchain: SwapchainKHR::null(),
				snapshot: None,
			},
			images: Vec::new(),
			available: semaphore(device),
			rendered: semaphore(device),
		};
		this.make(surface, window);
		this
	}

	pub fn acquire(&self) -> (ExternalImage, u32) {
		unsafe {
			let (id, _) = self
				.swapchain_ext
				.acquire_next_image(self.swapchain, u64::MAX, self.available, Fence::null())
				.unwrap();
			(
				ExternalImage {
					handle: self.images[id as usize],
					prev_usage: ExternalSync {
						semaphore: self.available,
						value: 0,
						usage: &[ImageUsageType::Present],
					},
					next_usage: ExternalSync {
						semaphore: self.rendered,
						value: 0,
						usage: &[ImageUsageType::Present],
					},
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
						swapchain: SwapchainKHR::null(),
						snapshot: None,
					};
				}
			}

			self.swapchain_ext
				.queue_present(
					*device.graphics_queue(),
					&PresentInfoKHR::builder()
						.wait_semaphores(&[self.rendered])
						.swapchains(&[self.swapchain])
						.image_indices(&[id])
						.build(),
				)
				.unwrap();
		}
	}

	pub fn resize(&mut self, surface: SurfaceKHR, window: &Window, graph: &RenderGraph) {
		unsafe {
			self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
		}
		self.old_swapchain = OldSwapchain {
			swapchain: self.swapchain,
			snapshot: Some(graph.snapshot()),
		};
		self.make(surface, window);
	}

	fn make(&mut self, surface: SurfaceKHR, window: &Window) {
		unsafe {
			self.swapchain = self
				.swapchain_ext
				.create_swapchain(
					&SwapchainCreateInfoKHR::builder()
						.surface(surface)
						.min_image_count(2)
						.image_format(Format::B8G8R8A8_SRGB)
						.image_color_space(ColorSpaceKHR::SRGB_NONLINEAR)
						.image_extent(Extent2D {
							width: window.inner_size().width,
							height: window.inner_size().height,
						})
						.image_array_layers(1)
						.image_usage(ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST) // Check
						.image_sharing_mode(SharingMode::EXCLUSIVE)
						.pre_transform(SurfaceTransformFlagsKHR::IDENTITY)
						.composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
						.present_mode(PresentModeKHR::FIFO)
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
