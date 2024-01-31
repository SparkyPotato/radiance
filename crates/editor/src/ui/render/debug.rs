use egui::{Context, Ui, Window};
use radiance_core::CoreDevice;
use radiance_graph::alloc::AllocatorVisualizer;
use radiance_passes::mesh::visbuffer::Camera;

use crate::ui::render::camera::CameraController;

pub struct DebugWindows {
	pub ground_truth: bool,
	culling: bool,
	cull_camera: Option<Camera>,
	alloc: AllocatorVisualizer,
	alloc_breakdown: bool,
	alloc_block: bool,
}

impl DebugWindows {
	pub fn new() -> Self {
		Self {
			ground_truth: false,
			culling: false,
			cull_camera: None,
			alloc: AllocatorVisualizer::new(),
			alloc_breakdown: false,
			alloc_block: false,
		}
	}

	pub fn draw_menu(&mut self, ui: &mut Ui) {
		ui.checkbox(&mut self.ground_truth, "ground truth");
		ui.checkbox(&mut self.culling, "culling");
		ui.checkbox(&mut self.alloc_breakdown, "alloc breakdown");
		ui.checkbox(&mut self.alloc_block, "alloc blocks");
	}

	pub fn draw(&mut self, ctx: &Context, device: &CoreDevice, camera: &CameraController) {
		Window::new("culling").open(&mut self.culling).show(ctx, |ui| {
			let mut locked = self.cull_camera.is_some();
			ui.checkbox(&mut locked, "lock culling");
			if self.cull_camera.is_none() && locked {
				self.cull_camera = Some(camera.get());
			} else if !locked {
				self.cull_camera = None;
			}
		});

		let alloc = device.allocator();
		self.alloc
			.render_breakdown_window(ctx, &alloc, &mut self.alloc_breakdown);
		egui::Window::new("Allocator Memory Blocks")
			.open(&mut self.alloc_block)
			.show(ctx, |ui| self.alloc.render_memory_block_ui(ui, &alloc));
		self.alloc.render_memory_block_visualization_windows(ctx, &alloc);
	}

	pub fn cull_camera(&self) -> Option<Camera> { self.cull_camera }
}
