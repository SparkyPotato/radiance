use ash::{
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
		SemaphoreCreateInfo,
		SharingMode,
		SurfaceKHR,
		SwapchainCreateInfoKHR,
		SwapchainKHR,
	},
};
use radiance_graph::{
	device::Device,
	graph::{ExecutionSnapshot, ExternalImage, ExternalSync, ImageUsageType, RenderGraph},
	Result,
};

struct OldSwapchain {
	swapchain: SwapchainKHR,
	snapshot: Option<ExecutionSnapshot>,
}

pub struct Window {
	pub window: winit::window::Window,
	surface: SurfaceKHR,
	swapchain_ext: khr::Swapchain,
	old_swapchain: OldSwapchain,
	swapchain: SwapchainKHR,
	images: Vec<Image>,
	available: Semaphore,
	rendered: Semaphore,
	format: Format,
}

impl Window {
	pub fn new(device: &Device, window: winit::window::Window, surface: SurfaceKHR) -> Result<Self> {
		let swapchain_ext = khr::Swapchain::new(device.instance(), device.device());
		let mut this = Self {
			window,
			surface,
			swapchain_ext,
			swapchain: SwapchainKHR::null(),
			old_swapchain: OldSwapchain {
				swapchain: SwapchainKHR::null(),
				snapshot: None,
			},
			images: Vec::new(),
			available: semaphore(device),
			rendered: semaphore(device),
			format: Format::UNDEFINED,
		};
		this.make(device)?;
		Ok(this)
	}

	pub fn request_redraw(&self) { self.window.request_redraw(); }

	pub fn acquire(&self) -> Result<(ExternalImage, u32)> {
		unsafe {
			let (id, _) =
				self.swapchain_ext
					.acquire_next_image(self.swapchain, u64::MAX, self.available, Fence::null())?;
			Ok((
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
			))
		}
	}

	pub fn present(&mut self, device: &Device, id: u32, graph: &RenderGraph) -> Result<()> {
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

			self.swapchain_ext.queue_present(
				*device.graphics_queue(),
				&PresentInfoKHR::builder()
					.wait_semaphores(&[self.rendered])
					.swapchains(&[self.swapchain])
					.image_indices(&[id])
					.build(),
			)?;

			Ok(())
		}
	}

	pub fn format(&self) -> Format { self.format }

	pub fn resize(&mut self, device: &Device, graph: &RenderGraph) -> Result<()> {
		if let Some(snapshot) = self.old_swapchain.snapshot.take() {
			snapshot.wait(device)?;
			unsafe {
				self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			}
		}

		self.old_swapchain = OldSwapchain {
			swapchain: self.swapchain,
			snapshot: Some(graph.snapshot()),
		};
		self.make(device)
	}

	pub fn destroy(&mut self, device: &Device) {
		unsafe {
			self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			self.swapchain_ext.destroy_swapchain(self.swapchain, None);
			device.surface_ext().unwrap().destroy_surface(self.surface, None);
			device.device().destroy_semaphore(self.available, None);
			device.device().destroy_semaphore(self.rendered, None);
		}
	}

	fn make(&mut self, device: &Device) -> Result<()> {
		unsafe {
			let surface_ext = device.surface_ext().unwrap();
			let capabilities =
				surface_ext.get_physical_device_surface_capabilities(device.physical_device(), self.surface)?;
			let formats = surface_ext.get_physical_device_surface_formats(device.physical_device(), self.surface)?;

			let (format, color_space) = formats
				.iter()
				.find(|format| {
					format.format == Format::B8G8R8A8_SRGB && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
				})
				.map(|format| (format.format, format.color_space))
				.unwrap_or((formats[0].format, formats[0].color_space));

			self.swapchain = self
				.swapchain_ext
				.create_swapchain(
					&SwapchainCreateInfoKHR::builder()
						.surface(self.surface)
						.min_image_count(if (capabilities.min_image_count..=capabilities.max_image_count).contains(&2) { 2 } else { capabilities.min_image_count })
						.image_format(format)
						.image_color_space(color_space)
						.image_extent(Extent2D {
							width: self.window.inner_size().width,
							height: self.window.inner_size().height,
						})
						.image_array_layers(1)
						.image_usage(ImageUsageFlags::COLOR_ATTACHMENT) // Check
						.image_sharing_mode(SharingMode::EXCLUSIVE)
						.pre_transform(capabilities.current_transform)
						.composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
						.present_mode(PresentModeKHR::FIFO)
						.old_swapchain(self.old_swapchain.swapchain)
						.clipped(true),
					None,
				)
				.unwrap();
			self.images = self.swapchain_ext.get_swapchain_images(self.swapchain).unwrap();
			self.format = format;
		}

		Ok(())
	}
}

pub fn semaphore(device: &Device) -> Semaphore {
	unsafe {
		device
			.device()
			.create_semaphore(&SemaphoreCreateInfo::builder().build(), None)
			.unwrap()
	}
}
