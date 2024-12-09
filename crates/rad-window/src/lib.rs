use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{
	ash::{khr, vk},
	device::{Device, Graphics, SyncPoint},
	graph::SwapchainImage,
	Result,
};
use winit::{application::ApplicationHandler, dpi::LogicalSize, event_loop::EventLoop, window::WindowId};
pub use winit::{event::WindowEvent, event_loop::ActiveEventLoop, window::Window as WinitWindow};

pub struct WindowModule;

impl Module for WindowModule {
	fn init(_: &mut EngineBuilder) {}
}

pub fn run(app: impl App) -> Result<()> {
	let event_loop = EventLoop::new().map_err(|x| x.to_string())?;
	let mut app = AppWrapper {
		app,
		minimized: false,
		window: None,
	};
	event_loop.run_app(&mut app).map_err(|x| x.to_string().into())
}

struct OldSwapchain {
	swapchain: vk::SwapchainKHR,
	sync: SyncPoint<Graphics>,
}

struct AppWrapper<T: App> {
	app: T,
	minimized: bool,
	window: Option<Window>,
}

impl<T: App> ApplicationHandler for AppWrapper<T> {
	fn resumed(&mut self, el: &ActiveEventLoop) {
		self.window = Some(Window::new("radiance", el).unwrap());
		self.app.init(el, &self.window.as_ref().unwrap().inner).unwrap();
	}

	fn suspended(&mut self, _: &ActiveEventLoop) { self.window.as_mut().unwrap().destroy(); }

	fn about_to_wait(&mut self, _: &ActiveEventLoop) {
		if !self.minimized {
			self.window.as_mut().unwrap().inner.request_redraw();
		}
	}

	fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
		match event {
			WindowEvent::RedrawRequested => {
				let window = self.window.as_mut().unwrap();
				let Some((image, id)) = window.acquire().unwrap() else {
					self.minimized = true;
					return;
				};
				self.app.draw(&window.inner, image).unwrap();
				let _ = window.present(id);
			},
			WindowEvent::Resized(x) => {
				if x.width != 0 && x.height != 0 {
					self.minimized = false;
				}
				self.window.as_mut().unwrap().resize().unwrap()
			},
			WindowEvent::CloseRequested => el.exit(),
			x => self.app.event(&self.window.as_ref().unwrap().inner, x).unwrap(),
		}
	}
}

struct Window {
	inner: WinitWindow,
	surface: vk::SurfaceKHR,
	swapchain_ext: khr::swapchain::Device,
	old_swapchain: OldSwapchain,
	swapchain: vk::SwapchainKHR,
	images: Vec<vk::Image>,
	semas: [(vk::Semaphore, vk::Semaphore); 3],
	curr_frame: usize,
	format: vk::Format,
	size: vk::Extent2D,
}

impl Window {
	pub fn new(title: &str, event_loop: &ActiveEventLoop) -> Result<Self> {
		let device: &Device = Engine::get().global();
		let inner = event_loop
			.create_window(
				winit::window::Window::default_attributes()
					.with_title(title)
					.with_inner_size(LogicalSize::new(1280, 720)),
			)
			.unwrap();
		let surface = unsafe { device.create_surface(&inner, event_loop).unwrap() };

		let swapchain_ext = khr::swapchain::Device::new(device.instance(), device.device());
		let mut this = Self {
			inner,
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

	fn acquire(&mut self) -> Result<Option<(SwapchainImage, u32)>> {
		unsafe {
			self.curr_frame = (self.curr_frame + 1) % 3;
			let (available, rendered) = self.semas[self.curr_frame];
			let (id, _) =
				match self
					.swapchain_ext
					.acquire_next_image(self.swapchain, u64::MAX, available, vk::Fence::null())
				{
					Ok(x) => x,
					Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(None),
					Err(x) => return Err(x.into()),
				};
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

			Ok(Some(ret))
		}
	}

	fn present(&mut self, id: u32) -> Result<()> {
		unsafe {
			tracing::trace!("present");

			let device: &Device = Engine::get().global();
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

	fn resize(&mut self) -> Result<()> {
		let device: &Device = Engine::get().global();
		self.make(device)
	}

	fn destroy(&mut self) {
		unsafe {
			let device: &Device = Engine::get().global();
			self.swapchain_ext.destroy_swapchain(self.old_swapchain.swapchain, None);
			self.swapchain_ext.destroy_swapchain(self.swapchain, None);
			device.surface_ext().destroy_surface(self.surface, None);
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
			let size = self.inner.inner_size();
			if size.height == 0 || size.width == 0 {
				return Ok(());
			}

			let surface_ext = device.surface_ext();
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

pub trait App {
	fn init(&mut self, el: &ActiveEventLoop, window: &WinitWindow) -> Result<()>;

	fn draw(&mut self, window: &WinitWindow, image: SwapchainImage) -> Result<()>;

	fn event(&mut self, window: &WinitWindow, event: WindowEvent) -> Result<()>;
}
