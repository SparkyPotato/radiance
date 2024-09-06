use ash::{khr, vk};
use radiance_graph::{
	device::{Device, Graphics, SyncPoint},
	graph::{SwapchainImage, FRAMES_IN_FLIGHT},
	Result,
};

struct OldSwapchain {
	swapchain: vk::SwapchainKHR,
	sync: SyncPoint<Graphics>,
}

pub struct Window {
	pub window: winit::window::Window,
	surface: vk::SurfaceKHR,
	swapchain_ext: khr::swapchain::Device,
	old_swapchain: OldSwapchain,
	swapchain: vk::SwapchainKHR,
	images: Vec<vk::Image>,
	semas: [(vk::Semaphore, vk::Semaphore); FRAMES_IN_FLIGHT + 1],
	curr_frame: usize,
	format: vk::Format,
	size: vk::Extent2D,
}

impl Window {
	pub fn new(device: &Device, window: winit::window::Window, surface: vk::SurfaceKHR) -> Result<Self> {
		let swapchain_ext = khr::swapchain::Device::new(device.instance(), device.device());
		let mut this = Self {
			window,
			surface,
			swapchain_ext,
			swapchain: vk::SwapchainKHR::null(),
			old_swapchain: OldSwapchain {
				swapchain: vk::SwapchainKHR::null(),
				sync: SyncPoint::default(),
			},
			images: Vec::new(),
			semas: [
				(semaphore(device)?, semaphore(device)?),
				(semaphore(device)?, semaphore(device)?),
				(semaphore(device)?, semaphore(device)?),
			],
			curr_frame: 0,
			format: vk::Format::UNDEFINED,
			size: vk::Extent2D::default(),
		};
		this.make(device)?;
		Ok(this)
	}

	pub fn request_redraw(&self) { self.window.request_redraw(); }

	pub fn acquire(&mut self) -> Result<(SwapchainImage, u32)> {
		unsafe {
			self.curr_frame = (self.curr_frame + 1) % 3;
			let (available, rendered) = self.semas[self.curr_frame];
			let (id, _) =
				self.swapchain_ext
					.acquire_next_image(self.swapchain, u64::MAX, available, vk::Fence::null())?;
			let ret = (
				SwapchainImage {
					handle: self.images[id as usize],
					size: self.size,
					format: self.format,
					available,
					rendered,
				},
				id,
			);

			Ok(ret)
		}
	}

	pub fn present(&mut self, device: &Device, id: u32) -> Result<()> {
		unsafe {
			self.cleanup_old(device)?;

			let (_, rendered) = self.semas[self.curr_frame];
			self.swapchain_ext.queue_present(
				*device.queue::<Graphics>(),
				&vk::PresentInfoKHR::default()
					.wait_semaphores(&[rendered])
					.swapchains(&[self.swapchain])
					.image_indices(&[id]),
			)?;

			Ok(())
		}
	}

	pub fn resize(&mut self, device: &Device) -> Result<()> { self.make(device) }

	pub fn destroy(&mut self, device: &Device) {
		unsafe {
			self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			self.swapchain_ext.destroy_swapchain(self.swapchain, None);
			device.surface_ext().unwrap().destroy_surface(self.surface, None);
			for (available, rendered) in self.semas {
				device.device().destroy_semaphore(available, None);
				device.device().destroy_semaphore(rendered, None);
			}
		}
	}

	fn cleanup_old(&mut self, device: &Device) -> Result<()> {
		if self.old_swapchain.swapchain != vk::SwapchainKHR::null() {
			self.old_swapchain.sync.wait(device)?;
			unsafe {
				self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			}
			self.old_swapchain.swapchain = vk::SwapchainKHR::null();
		}

		Ok(())
	}

	fn make(&mut self, device: &Device) -> Result<()> {
		unsafe {
			let size = self.window.inner_size();
			if size.height == 0 || size.width == 0 {
				return Ok(());
			}

			let surface_ext = device.surface_ext().unwrap();
			let capabilities =
				surface_ext.get_physical_device_surface_capabilities(device.physical_device(), self.surface)?;
			let formats = surface_ext.get_physical_device_surface_formats(device.physical_device(), self.surface)?;

			let (format, color_space) = formats
				.iter()
				.find(|format| {
					format.format == vk::Format::B8G8R8A8_UNORM
						&& format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
				})
				.map(|format| (format.format, format.color_space))
				.unwrap_or((formats[0].format, formats[0].color_space));

			self.cleanup_old(device)?;
			if self.swapchain != vk::SwapchainKHR::null() {
				self.old_swapchain = OldSwapchain {
					swapchain: self.swapchain,
					sync: device.current_sync_point(),
				};
			}

			self.swapchain = self
				.swapchain_ext
				.create_swapchain(
					&vk::SwapchainCreateInfoKHR::default()
						.surface(self.surface)
						.min_image_count(if (capabilities.min_image_count..=capabilities.max_image_count).contains(&2) { 2 } else { capabilities.min_image_count })
						.image_format(format)
						.image_color_space(color_space)
						.image_extent(vk::Extent2D {
							width: size.width,
							height: size.height,
						})
						.image_array_layers(1)
						.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // Check
						.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
						.pre_transform(capabilities.current_transform)
						.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
						.present_mode(vk::PresentModeKHR::FIFO)
						.old_swapchain(self.old_swapchain.swapchain)
						.clipped(true),
					None,
				)
				.unwrap();
			self.images = self.swapchain_ext.get_swapchain_images(self.swapchain).unwrap();
			self.format = format;
			self.size = vk::Extent2D {
				width: size.width,
				height: size.height,
			};
		}

		Ok(())
	}
}

pub fn semaphore(device: &Device) -> Result<vk::Semaphore> {
	unsafe {
		device
			.device()
			.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
			.map_err(Into::into)
	}
}
