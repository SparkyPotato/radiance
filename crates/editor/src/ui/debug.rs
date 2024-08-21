use egui::{ComboBox, Context, DragValue, Window};
use radiance_graph::{alloc::AllocatorVisualizer, device::Device};
use radiance_passes::debug::mesh::DebugVis;

pub struct Debug {
	pub enabled: bool,
	debug_vis: DebugVis,
	bottom: u32,
	top: u32,
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
			bottom: 0,
			top: 1000,
			alloc: AllocatorVisualizer::new(),
			alloc_breakdown: false,
			alloc_block: false,
			arena_size: 0,
		}
	}

	fn vis_to_index(vis: DebugVis) -> usize {
		match vis {
			DebugVis::Triangles => 0,
			DebugVis::Meshlets => 1,
			DebugVis::Overdraw(..) => 2,
		}
	}

	fn text_of_index(vis: usize) -> &'static str {
		match vis {
			0 => "triangles",
			1 => "meshlets",
			2 => "overdraw",
			_ => unreachable!(),
		}
	}

	pub fn render(&mut self, device: &Device, ctx: &Context) {
		Window::new("debug").open(&mut self.enabled).show(ctx, |ui| {
			let mut sel = Self::vis_to_index(self.debug_vis);
			ComboBox::from_label("debug vis")
				.selected_text(Self::text_of_index(sel))
				.show_index(ui, &mut sel, 3, Self::text_of_index);
			self.debug_vis = match sel {
				0 => DebugVis::Triangles,
				1 => DebugVis::Meshlets,
				2 => DebugVis::Overdraw(self.bottom, self.top),
				_ => unreachable!(),
			};

			match &mut self.debug_vis {
				DebugVis::Overdraw(b, t) => {
					ui.horizontal(|ui| {
						ui.add(DragValue::new(&mut self.bottom).speed(0.2));
						ui.add(DragValue::new(&mut self.top).speed(0.2));
					});
					*b = self.bottom;
					*t = self.top;
				},
				_ => {},
			}

			ui.checkbox(&mut self.alloc_breakdown, "gpu alloc breakdown");
			ui.checkbox(&mut self.alloc_block, "gpu alloc blocks");

			ui.label(format!(
				"arena size: {:.2} MB",
				self.arena_size as f32 / (1024.0 * 1024.0)
			));
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

	pub fn debug_vis(&self) -> DebugVis { self.debug_vis }
}
