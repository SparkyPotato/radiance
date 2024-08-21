use egui::{ComboBox, Context, Window};
use radiance_graph::{alloc::AllocatorVisualizer, device::Device};
use radiance_passes::debug::mesh::DebugVis;

pub struct Debug {
	pub enabled: bool,
	debug_vis: DebugVis,
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
			alloc: AllocatorVisualizer::new(),
			alloc_breakdown: false,
			alloc_block: false,
			arena_size: 0,
		}
	}

	fn text_of_vis(vis: DebugVis) -> &'static str {
		match vis {
			DebugVis::Triangles => "triangles",
			DebugVis::Meshlets => "meshlets",
		}
	}

	fn vis_from_usize(val: usize) -> DebugVis {
		match val {
			0 => DebugVis::Triangles,
			1 => DebugVis::Meshlets,
			_ => unreachable!(),
		}
	}

	pub fn render(&mut self, device: &Device, ctx: &Context) {
		Window::new("debug").open(&mut self.enabled).show(ctx, |ui| {
			let mut sel = self.debug_vis as usize;
			ComboBox::from_label("debug vis")
				.selected_text(Self::text_of_vis(self.debug_vis))
				.show_index(ui, &mut sel, 2, |x| Self::text_of_vis(Self::vis_from_usize(x)));
			self.debug_vis = Self::vis_from_usize(sel);

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
