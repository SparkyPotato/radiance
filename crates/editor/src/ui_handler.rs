use egui::Context;
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::ScreenDescriptor;
use radiance_graph::Result;
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
	pub fn new(device: &CoreDevice, core: &RenderCore, event_loop: &EventLoop<()>, window: &Window) -> Result<Self> {
		let ctx = Context::default();
		let (defs, fonts) = Fonts::defs();
		ctx.set_fonts(defs);

		Ok(Self {
			ctx,
			platform_state: egui_winit::State::new(event_loop),
			renderer: radiance_egui::Renderer::new(device, core, window.format())?,
			fonts,
		})
	}

	pub fn fonts(&self) -> &Fonts { &self.fonts }

	pub fn begin_frame(&mut self, window: &Window) {
		self.ctx
			.begin_frame(self.platform_state.take_egui_input(&window.window));
	}

	pub fn run<'pass>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, window: &Window,
	) -> Result<u32> {
		let (image, id) = {
			tracy::zone!("swapchain acquire");
			window.acquire()?
		};

		let output = self.ctx.end_frame();

		{
			tracy::zone!("handle window output");
			self.platform_state
				.handle_platform_output(&window.window, &self.ctx, output.platform_output);
		}

		let tris = {
			tracy::zone!("tessellate shapes");
			self.ctx.tessellate(output.shapes)
		};

		self.renderer.run(
			device,
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

	pub fn on_event(&mut self, event: &WindowEvent) { let _ = self.platform_state.on_event(&self.ctx, event); }

	pub unsafe fn destroy(self, device: &CoreDevice) { self.renderer.destroy(device); }
}
