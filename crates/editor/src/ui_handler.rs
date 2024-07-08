use egui::{Context, ViewportId};
use egui_winit::pixels_per_point;
use radiance_egui::ScreenDescriptor;
use radiance_graph::{device::Device, graph::Frame, Result};
use vek::Vec2;
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::{ui::Fonts, window::Window};

pub struct UiHandler {
	pub ctx: Context,
	platform_state: egui_winit::State,
	renderer: radiance_egui::Renderer,
	fonts: Fonts,
}

impl UiHandler {
	pub fn new(device: &Device, event_loop: &EventLoop<()>, window: &Window) -> Result<Self> {
		let ctx = Context::default();
		let (defs, fonts) = Fonts::defs();
		ctx.set_fonts(defs);

		let platform_state = egui_winit::State::new(
			ctx.clone(),
			ViewportId::default(),
			event_loop,
			Some(window.window.scale_factor() as _),
			None,
		);

		Ok(Self {
			ctx,
			platform_state,
			renderer: radiance_egui::Renderer::new(device)?,
			fonts,
		})
	}

	pub fn fonts(&self) -> &Fonts { &self.fonts }

	pub fn begin_frame(&mut self, window: &Window) {
		self.ctx
			.begin_frame(self.platform_state.take_egui_input(&window.window));
	}

	pub fn run<'pass, 'graph>(&'pass mut self, frame: &mut Frame<'pass, 'graph>, window: &mut Window) -> Result<u32>
	where
		'graph: 'pass,
	{
		let output = self.ctx.end_frame();

		{
			tracy::zone!("handle window output");
			self.platform_state
				.handle_platform_output(&window.window, output.platform_output);
		}

		let tris = {
			tracy::zone!("tessellate shapes");
			self.ctx
				.tessellate(output.shapes, pixels_per_point(&self.ctx, &window.window))
		};

		let (image, id) = {
			tracy::zone!("swapchain acquire");
			window.acquire()?
		};
		self.renderer.run(
			frame,
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

	pub fn on_event(&mut self, window: &Window, event: &WindowEvent) {
		let _ = self.platform_state.on_window_event(&window.window, event);
	}

	pub unsafe fn destroy(self, device: &Device) { self.renderer.destroy(device); }
}
