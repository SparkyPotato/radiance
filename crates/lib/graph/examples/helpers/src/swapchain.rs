use radiance_graph::{
	ash::{extensions::khr, vk},
	device::{Device, Graphics, QueueWait, SyncPoint, SyncStage},
	graph::{ExternalImage, ImageUsage, ImageUsageType, PassBuilder, Res, Signal, Wait},
	resource::ImageView,
};
use winit::window::Window;

use crate::misc::semaphore;

struct OldSwapchain {
	swapchain: vk::SwapchainKHR,
	sync: Option<SyncPoint<Graphics>>,
}

pub struct Swapchain {
	surface: vk::SurfaceKHR,
	swapchain_ext: khr::Swapchain,
	swapchain: vk::SwapchainKHR,
	old_swapchain: OldSwapchain,
	images: Vec<vk::Image>,
	available: vk::Semaphore,
	rendered: vk::Semaphore,
	size: vk::Extent3D,
}

#[derive(Copy, Clone)]
pub struct SwapchainImage(u32);

impl Swapchain {
	pub fn new(device: &Device, surface: vk::SurfaceKHR, window: &Window) -> Self {
		let swapchain_ext = khr::Swapchain::new(device.instance(), device.device());
		let mut this = Self {
			surface,
			swapchain_ext,
			swapchain: vk::SwapchainKHR::null(),
			old_swapchain: OldSwapchain {
				swapchain: vk::SwapchainKHR::null(),
				sync: None,
			},
			images: Vec::new(),
			available: semaphore(device),
			rendered: semaphore(device),
			size: vk::Extent3D {
				width: window.inner_size().width,
				height: window.inner_size().height,
				depth: 1,
			},
		};
		this.make(window);
		this
	}

	pub fn acquire(&self) -> SwapchainImage {
		unsafe {
			let (id, _) = self
				.swapchain_ext
				.acquire_next_image(self.swapchain, u64::MAX, self.available, vk::Fence::null())
				.unwrap();
			SwapchainImage(id)
		}
	}

	pub fn import_image<C>(
		&self, image: SwapchainImage, usage: ImageUsage, pass: &mut PassBuilder<C>,
	) -> Res<ImageView> {
		// TODO: this is too goofy.
		pass.output(
			ExternalImage {
				handle: self.images[image.0 as usize],
				size: self.size,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				wait: Wait {
					usage: ImageUsage {
						usages: &[ImageUsageType::Present],
						..Default::default()
					},
					wait: QueueWait {
						binary_semaphores: &[SyncStage {
							point: self.available,
							stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
						}],
						..Default::default()
					},
				},
				signal: Signal {
					usage: ImageUsage {
						usages: &[ImageUsageType::Present],
						..Default::default()
					},
					signal: &[SyncStage {
						point: self.rendered,
						stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
					}],
				},
			},
			usage,
		)
	}

	pub fn present(&mut self, device: &Device, image: SwapchainImage) {
		unsafe {
			if let Some(sync) = self.old_swapchain.sync {
				if sync.is_complete(device).unwrap_or_else(|_| false) {
					self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
					self.old_swapchain = OldSwapchain {
						swapchain: vk::SwapchainKHR::null(),
						sync: None,
					};
				}
			}

			self.swapchain_ext
				.queue_present(
					*device.queue::<Graphics>(),
					&vk::PresentInfoKHR::builder()
						.wait_semaphores(&[self.rendered])
						.swapchains(&[self.swapchain])
						.image_indices(&[image.0])
						.build(),
				)
				.unwrap();
		}
	}

	pub fn resize(&mut self, device: &Device, window: &Window) {
		if let Some(sync) = self.old_swapchain.sync.take() {
			sync.wait(device).unwrap();
			unsafe {
				self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			}
		}

		self.old_swapchain = OldSwapchain {
			swapchain: self.swapchain,
			sync: Some(device.current_sync_point()),
		};
		self.make(window);
		self.size = vk::Extent3D {
			width: window.inner_size().width,
			height: window.inner_size().height,
			depth: 1,
		};
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
