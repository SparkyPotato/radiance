use std::sync::Arc;

use egui::{Context, FontData, FontDefinitions, FontFamily};
use radiance_egui::ScreenDescriptor;
use radiance_graph::{device::Device, graph::Frame, Result};
use vek::Vec2;
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::window::Window;

pub struct UiHandler {
	ctx: Context,
	platform_state: egui_winit::State,
	renderer: radiance_egui::Renderer,
	icon: Arc<str>,
}

impl UiHandler {
	pub fn new(device: &Device, event_loop: &EventLoop<()>, window: &Window) -> Result<Self> {
		let mut ctx = Context::default();
		let mut fonts = FontDefinitions::empty();
		fonts.font_data.insert(
			"Inter".to_string(),
			FontData::from_static(include_bytes!("../fonts/Inter/Inter-Regular.otf")),
		);
		fonts.font_data.insert(
			"Font Awesome".to_string(),
			FontData::from_static(include_bytes!(
				"../fonts/Font Awesome/Font Awesome 6 Free-Solid-900.otf"
			)),
		);
		let icon: Arc<str> = Arc::from("Icon");
		fonts
			.families
			.insert(FontFamily::Proportional, vec!["Inter".to_string()]);
		fonts
			.families
			.insert(FontFamily::Name(icon.clone()), vec!["Font Awesome".to_string()]);
		ctx.set_fonts(fonts);

		Ok(Self {
			ctx,
			platform_state: egui_winit::State::new(event_loop),
			renderer: radiance_egui::Renderer::new(device, window.format())?,
			icon,
		})
	}

	pub fn icon(&self) -> &Arc<str> { &self.icon }

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
