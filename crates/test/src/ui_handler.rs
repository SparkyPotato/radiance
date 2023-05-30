use egui::Context;
use radiance_egui::ScreenDescriptor;
use radiance_graph::{device::Device, graph::Frame, Result};
use vek::Vec2;
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::window::Window;

pub struct UiHandler {
	ctx: Context,
	platform_state: egui_winit::State,
	renderer: radiance_egui::Renderer,
}

impl UiHandler {
	pub fn new(device: &Device, event_loop: &EventLoop<()>, window: &Window) -> Result<Self> {
		Ok(Self {
			ctx: Context::default(),
			platform_state: egui_winit::State::new(event_loop),
			renderer: radiance_egui::Renderer::new(device, window.format())?,
		})
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, device: &Device, window: &Window, run: impl FnOnce(&Context),
	) -> Result<u32> {
		let (image, id) = {
			tracy::zone!("swapchain acquire");
			window.acquire()?
		};

		let output = self.ctx.run(self.platform_state.take_egui_input(&window.window), run);

		{
			tracy::zone!("handle window output");
			self.platform_state
				.handle_platform_output(&window.window, &self.ctx, output.platform_output);
		}

		let tris = {
			tracy::zone!("tessellate shapes");
			self.ctx.tessellate(output.shapes)
		};

		self.renderer.render(
			frame,
			device,
			tris,
			output.textures_delta,
			ScreenDescriptor {
				physical_size: Vec2::new(window.window.inner_size().width, window.window.inner_size().height),
				scaling: window.window.scale_factor() as _,
			},
			image,
		);

		Ok(id)
	}

	pub fn on_event(&mut self, event: &WindowEvent) { let _ = self.platform_state.on_event(&self.ctx, event); }

	pub unsafe fn destroy(self, device: &Device) { self.renderer.destroy(device); }
}
