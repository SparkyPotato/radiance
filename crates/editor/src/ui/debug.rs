use egui::{ComboBox, Context, DragValue, Ui, Window};
use radiance_graph::{
	alloc::AllocatorVisualizer,
	device::{Device, HotreloadStatus},
};
use radiance_passes::{
	debug::mesh::DebugVis,
	mesh::{CullStats, PassStats},
};

pub struct Debug {
	pub enabled: bool,
	debug_vis: DebugVis,
	scale: f32,
	alloc: AllocatorVisualizer,
	alloc_breakdown: bool,
	alloc_block: bool,
	arena_size: usize,
}

impl Debug {
	pub fn new() -> Self {
		Self {
			enabled: false,
			debug_vis: DebugVis::Meshlets,
			scale: 0.15,
			alloc: AllocatorVisualizer::new(),
			alloc_breakdown: false,
			alloc_block: false,
			arena_size: 0,
		}
	}

	fn text_of_index(vis: usize) -> &'static str {
		match vis {
			0 => "triangles",
			1 => "meshlets",
			2 => "overdraw",
			3 => "hw/sw",
			4 => "normals",
			5 => "uvs",
			6 => "error",
			_ => unreachable!(),
		}
	}

	pub fn render(&mut self, device: &Device, ctx: &Context, stats: CullStats) {
		Window::new("debug").open(&mut self.enabled).show(ctx, |ui| {
			let mut sel = self.debug_vis.to_u32() as usize;
			ComboBox::from_label("debug vis")
				.selected_text(Self::text_of_index(sel))
				.show_index(ui, &mut sel, 7, Self::text_of_index);
			self.debug_vis = match sel {
				0 => DebugVis::Triangles,
				1 => DebugVis::Meshlets,
				2 => DebugVis::Overdraw(self.scale),
				3 => DebugVis::HwSw,
				4 => DebugVis::Normals,
				5 => DebugVis::Uvs,
				6 => DebugVis::Error,
				_ => unreachable!(),
			};

			match &mut self.debug_vis {
				DebugVis::Overdraw(s) => {
					ui.horizontal(|ui| {
						ui.add(DragValue::new(&mut self.scale).speed(0.01).clamp_range(0.0..=1.0));
					});
					*s = self.scale;
				},
				_ => {},
			}

			ui.checkbox(&mut self.alloc_breakdown, "gpu alloc breakdown");
			ui.checkbox(&mut self.alloc_block, "gpu alloc blocks");

			ui.label(format!(
				"arena size: {:.2} MB",
				self.arena_size as f32 / (1024.0 * 1024.0)
			));

			ui.horizontal(|ui| {
				ui.label("hotreload: ");
				match device.hotreload_status() {
					HotreloadStatus::Waiting => ui.label("ready"),
					HotreloadStatus::Recompiling => ui.spinner(),
					HotreloadStatus::Errored => ui.label("errored"),
				}
			});

			ui.label("early");
			Self::pass_stats(ui, stats.early);
			ui.label("late");
			Self::pass_stats(ui, stats.late);
		});

		let alloc = device.allocator();
		self.alloc
			.render_breakdown_window(ctx, &alloc, &mut self.alloc_breakdown);
		Window::new("Allocator Memory Blocks")
			.open(&mut self.alloc_block)
			.show(ctx, |ui| self.alloc.render_memory_block_ui(ui, &alloc));
		self.alloc.render_memory_block_visualization_windows(ctx, &alloc);
	}

	fn pass_stats(ui: &mut Ui, pass: PassStats) {
		ui.label(format!("instances: {}", pass.instances));
		ui.label(format!("candidate meshlets: {}", pass.candidate_meshlets));
		ui.label(format!("hw meshlets: {}", pass.hw_meshlets));
		ui.label(format!("sw meshlets: {}", pass.sw_meshlets));
	}

	pub fn set_arena_size(&mut self, size: usize) { self.arena_size = size; }

	pub fn debug_vis(&self) -> DebugVis { self.debug_vis }
}
