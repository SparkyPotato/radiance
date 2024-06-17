use egui::{ComboBox, Context, Window};
use radiance_core::CoreDevice;
use radiance_graph::alloc::AllocatorVisualizer;
use radiance_passes::mesh::visbuffer::Camera;

#[derive(Copy, Clone)]
pub enum RenderMode {
	Realtime,
	GroundTruth,
	CpuPath,
}

pub struct Debug {
	pub enabled: bool,
	render_mode: RenderMode,
	culling: bool,
	cull_camera: Option<Camera>,
	alloc: AllocatorVisualizer,
	alloc_breakdown: bool,
	alloc_block: bool,
	arena_size: usize,
}

impl Debug {
	pub fn new() -> Self {
		Self {
			enabled: false,
			render_mode: RenderMode::Realtime,
			culling: false,
			cull_camera: None,
			alloc: AllocatorVisualizer::new(),
			alloc_breakdown: false,
			alloc_block: false,
			arena_size: 0,
		}
	}

	pub fn render(&mut self, ctx: &Context, device: &CoreDevice) {
		Window::new("debug").open(&mut self.enabled).show(ctx, |ui| {
			let mut x = self.render_mode as usize;
			ComboBox::from_label("render mode").show_index(ui, &mut x, 3, |x| match x {
				0 => "Realtime",
				1 => "Ground Truth",
				2 => "CPU Path",
				_ => unreachable!(),
			});
			self.render_mode = match x {
				0 => RenderMode::Realtime,
				1 => RenderMode::GroundTruth,
				2 => RenderMode::CpuPath,
				_ => unreachable!(),
			};

			ui.checkbox(&mut self.alloc_breakdown, "gpu alloc breakdown");
			ui.checkbox(&mut self.alloc_block, "gpu alloc blocks");

			ui.label(format!(
				"arena size: {:.2} MB",
				self.arena_size as f32 / (1024.0 * 1024.0)
			));
		});

		Window::new("culling").open(&mut self.culling).show(ctx, |ui| {
			let mut locked = self.cull_camera.is_some();
			ui.checkbox(&mut locked, "lock culling");
			if self.cull_camera.is_none() && locked {
				// self.cull_camera = Some(camera.get());
			} else if !locked {
				self.cull_camera = None;
			}
		});

		let alloc = device.allocator();
		self.alloc
			.render_breakdown_window(ctx, &alloc, &mut self.alloc_breakdown);
		Window::new("Allocator Memory Blocks")
			.open(&mut self.alloc_block)
			.show(ctx, |ui| self.alloc.render_memory_block_ui(ui, &alloc));
		self.alloc.render_memory_block_visualization_windows(ctx, &alloc);
	}

	pub fn set_arena_size(&mut self, size: usize) { self.arena_size = size; }

	pub fn render_mode(&self) -> RenderMode { self.render_mode }

	pub fn cull_camera(&self) -> Option<Camera> { self.cull_camera }
}
