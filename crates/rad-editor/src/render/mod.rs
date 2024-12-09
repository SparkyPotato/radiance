use rad_core::Engine;
use rad_graph::{graph::Frame, Result};
use rad_renderer::{
	debug::mesh::{DebugMesh, DebugVis},
	mesh::{RenderInfo, VisBuffer},
	vek::Vec2,
};
use rad_ui::{
	egui::{CentralPanel, Context, Image, PointerButton, Sense},
	to_texture_id,
};
use rad_window::winit::{event::WindowEvent, window::Window};

use crate::{
	render::camera::{CameraController, Mode},
	world::WorldContext,
};

mod camera;

pub struct Renderer {
	visbuffer: VisBuffer,
	debug: DebugMesh,
	camera: CameraController,
	frame: u64,
}

impl Renderer {
	pub fn new() -> Result<Self> {
		let device = Engine::get().global();
		Ok(Self {
			visbuffer: VisBuffer::new(device)?,
			debug: DebugMesh::new(device)?,
			camera: CameraController::new(),
			frame: 0,
		})
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		self.camera.on_window_event(window, event);
	}

	pub fn render<'pass>(
		&'pass mut self, window: &Window, frame: &mut Frame<'pass, '_>, ctx: &Context, world: &'pass mut WorldContext,
	) {
		CentralPanel::default().show(ctx, |ui| {
			let rect = ui.available_rect_before_wrap();
			let size = rect.size();
			let resp = ui.allocate_rect(rect, Sense::click());

			if ctx.input(|x| resp.contains_pointer() && x.pointer.button_down(PointerButton::Secondary)) {
				self.camera.set_mode(window, Mode::Camera);
			} else {
				self.camera.set_mode(window, Mode::Default);
			}
			self.camera.control(ctx);
			self.camera.apply(world.editor_mut());
			world.edit_tick();

			let data = world.renderer().update(frame, self.frame);
			let visbuffer = self.visbuffer.run(
				frame,
				RenderInfo {
					data,
					size: Vec2::new(size.x as u32, size.y as u32),
					debug_info: DebugVis::Meshlets.requires_debug_info(),
				},
			);
			let img = self.debug.run(
				frame,
				DebugVis::Meshlets,
				visbuffer,
				// self.picker.get_sel().into_iter(),
				[].into_iter(),
			);
			ui.put(rect, Image::new((to_texture_id(img), size)));
		});

		self.frame += 1;
	}
}
