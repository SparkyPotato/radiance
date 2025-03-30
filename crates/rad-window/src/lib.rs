use std::ops::Deref;

use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{
	ash::{khr, vk},
	device::{Device, Graphics, Queues, SyncPoint},
	graph::SwapchainImage,
	Result,
};
pub use winit;
use winit::{
	application::ApplicationHandler,
	dpi::LogicalSize,
	event::WindowEvent,
	event_loop::{ActiveEventLoop, EventLoop},
	window::{Window as WinitWindow, WindowId},
};

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
		self.app.init(el, self.window.as_mut().unwrap()).unwrap();
	}

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
				self.app.draw(window, image).unwrap();
				let _ = window.present(id);
			},
			WindowEvent::Resized(x) => {
				if x.width != 0 && x.height != 0 {
					self.minimized = false;
				}
				self.window.as_mut().unwrap().resize().unwrap()
			},
			WindowEvent::CloseRequested => el.exit(),
			x => self.app.event(self.window.as_mut().unwrap(), x).unwrap(),
		}
	}
}

pub struct Window {
	inner: WinitWindow,
	surface: vk::SurfaceKHR,
	swapchain_ext: khr::swapchain::Device,
	old_swapchain: OldSwapchain,
	swapchain: vk::SwapchainKHR,
	images: Vec<vk::Image>,
	semas: [(vk::Semaphore, vk::Semaphore); 2],
	curr_frame: usize,
	format: vk::Format,
	size: vk::Extent2D,
	vsync: bool,
	hdr_requested: bool,
	hdr_supported: bool,
}

impl Deref for Window {
	type Target = WinitWindow;

	fn deref(&self) -> &Self::Target { &self.inner }
}

impl Window {
	fn new(title: &str, event_loop: &ActiveEventLoop) -> Result<Self> {
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
			],
			curr_frame: 0,
			format: vk::Format::UNDEFINED,
			size: vk::Extent2D::default(),
			vsync: true,
			hdr_requested: true,
			hdr_supported: false,
		};
		this.make(device)?;
		Ok(this)
	}

	pub fn hdr_enabled(&self) -> bool { self.format == vk::Format::A2B10G10R10_UNORM_PACK32 }

	pub fn hdr_supported(&self) -> bool { self.hdr_supported }

	pub fn set_hdr(&mut self, hdr: bool) -> Result<()> {
		if self.hdr_requested != hdr {
			self.hdr_requested = hdr;
			self.resize()?;
		}
		Ok(())
	}

	pub fn vsync_enabled(&self) -> bool { self.vsync }

	pub fn set_vsync(&mut self, vsync: bool) -> Result<()> {
		if self.vsync != vsync {
			self.vsync = vsync;
			self.resize()?;
		}
		Ok(())
	}

	fn acquire(&mut self) -> Result<Option<(SwapchainImage, u32)>> {
		unsafe {
			self.curr_frame ^= 1;
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

			let hdr_format = formats.iter().find(|x| {
				x.format == vk::Format::A2B10G10R10_UNORM_PACK32 && x.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT
			});
			self.hdr_supported = hdr_format.is_some();

			let (format, color_space) = self
				.hdr_requested
				.then_some(hdr_format)
				.flatten()
				.or_else(|| {
					formats.iter().find(|x| {
						x.format == vk::Format::B8G8R8A8_UNORM && x.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
					})
				})
				.map(|x| (x.format, x.color_space))
				.unwrap_or((formats[0].format, formats[0].color_space));

			self.cleanup_old(device)?;
			if self.swapchain != vk::SwapchainKHR::null() {
				self.old_swapchain = OldSwapchain {
					swapchain: self.swapchain,
					sync: device.current_sync_point(),
				};
			}

			self.size = vk::Extent2D {
				width: size.width,
				height: size.height,
			};

			let info = vk::SwapchainCreateInfoKHR::default()
				.surface(self.surface)
				.min_image_count(3.clamp(capabilities.min_image_count, capabilities.max_image_count))
				.image_format(format)
				.image_color_space(color_space)
				.image_extent(self.size)
				.image_array_layers(1)
				.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
				.pre_transform(capabilities.current_transform)
				.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
				.present_mode(if self.vsync {
					vk::PresentModeKHR::FIFO
				} else {
					vk::PresentModeKHR::IMMEDIATE
				})
				.old_swapchain(self.old_swapchain.swapchain)
				.clipped(true);
			self.swapchain = match device.queue_families() {
				Queues::Multiple {
					graphics,
					compute,
					transfer,
				} => self.swapchain_ext.create_swapchain(
					&info
						.image_sharing_mode(vk::SharingMode::CONCURRENT)
						.queue_family_indices(&[graphics, compute, transfer]),
					None,
				),
				Queues::Single(_) => self
					.swapchain_ext
					.create_swapchain(&info.image_sharing_mode(vk::SharingMode::EXCLUSIVE), None),
			}
			.unwrap();
			self.images = self.swapchain_ext.get_swapchain_images(self.swapchain).unwrap();
			self.format = format;
		}

		Ok(())
	}
}

impl Drop for Window {
	fn drop(&mut self) {
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
	fn init(&mut self, el: &ActiveEventLoop, window: &mut Window) -> Result<()>;

	fn draw(&mut self, window: &mut Window, image: SwapchainImage) -> Result<()>;

	fn event(&mut self, window: &mut Window, event: WindowEvent) -> Result<()>;
}
