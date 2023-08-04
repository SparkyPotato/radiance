use egui::{Context, Ui, Window};
use radiance_passes::mesh::visbuffer::Camera;

use crate::ui::render::camera::CameraController;

pub struct DebugWindows {
	culling: bool,
	cull_camera: Option<Camera>,
}

impl DebugWindows {
	pub fn new() -> Self {
		Self {
			culling: false,
			cull_camera: None,
		}
	}

	pub fn draw_menu(&mut self, ui: &mut Ui) { ui.checkbox(&mut self.culling, "culling"); }

	pub fn draw(&mut self, ctx: &Context, camera: &CameraController) {
		Window::new("culling").open(&mut self.culling).show(ctx, |ui| {
			let mut locked = self.cull_camera.is_some();
			ui.checkbox(&mut locked, "lock culling");
			if self.cull_camera.is_none() && locked {
				self.cull_camera = Some(camera.get());
			} else if !locked {
				self.cull_camera = None;
			}
		});
	}

	pub fn cull_camera(&self) -> Option<Camera> { self.cull_camera }
}
