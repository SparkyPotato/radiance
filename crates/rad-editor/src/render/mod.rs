use rad_core::Engine;
use rad_graph::{graph::Frame, Result};
use rad_renderer::{
	debug::mesh::{DebugMesh, DebugVis},
	mesh::{RenderInfo, VisBuffer},
	vek::Vec2,
};
use rad_ui::{
	egui::{CentralPanel, Context, Image, Sense},
	to_texture_id,
};

use crate::world::WorldContext;

pub struct Renderer {
	visbuffer: VisBuffer,
	debug: DebugMesh,
	frame: u64,
}

impl Renderer {
	pub fn new() -> Result<Self> {
		let device = Engine::get().global();
		Ok(Self {
			visbuffer: VisBuffer::new(device)?,
			debug: DebugMesh::new(device)?,
			frame: 0,
		})
	}

	pub fn render<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, world: &'pass mut WorldContext) {
		world.edit_tick();

		CentralPanel::default().show(ctx, |ui| {
			let rect = ui.available_rect_before_wrap();
			let size = rect.size();
			// let resp = ui.allocate_rect(rect, Sense::click());
			ui.allocate_rect(rect, Sense::click());

			let data = world.renderer().update(frame, self.frame);

			let visbuffer = self.visbuffer.run(
				frame,
				RenderInfo {
					data,
					size: Vec2::new(size.x as u32, size.y as u32),
					// debug_info: debug.debug_vis().requires_debug_info(),
					debug_info: false,
				},
			);

			// let clicked = resp.clicked();
			let img = self.debug.run(
				frame,
				// debug.debug_vis(),
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
