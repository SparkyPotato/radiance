use egui_plot::{Bar, BarChart, HPlacement, Plot, VLine, VPlacement};
use rad_graph::device::{Device, HotreloadStatus};
use rad_renderer::{
	debug::mesh::DebugVis,
	mesh::{CullStats, PassStats},
	tonemap::exposure::{ExposureCalc, ExposureStats},
};
use rad_ui::egui::{ComboBox, Context, DragValue, Ui, Window};

#[derive(Copy, Clone)]
pub enum RenderMode {
	Path,
	Debug,
}

#[derive(Copy, Clone)]
pub enum Tonemap {
	Aces,
	TonyMcMapface,
}

pub struct DebugWindow {
	pub enabled: bool,
	render_mode: RenderMode,
	tonemap: Tonemap,
	debug_vis: DebugVis,
	scale: f32,
}

impl DebugWindow {
	pub fn new() -> Self {
		Self {
			enabled: false,
			render_mode: RenderMode::Path,
			tonemap: Tonemap::TonyMcMapface,
			debug_vis: DebugVis::Meshlets,
			scale: 0.15,
		}
	}

	fn vis_text(vis: usize) -> &'static str {
		match vis {
			0 => "triangles",
			1 => "meshlets",
			2 => "overdraw",
			3 => "hw/sw",
			4 => "normals",
			5 => "uvs",
			6 => "error",
			7 => "base color",
			8 => "roughness",
			9 => "metallic",
			10 => "emissive",
			_ => unreachable!(),
		}
	}

	fn mode_text(mode: usize) -> &'static str {
		match mode {
			0 => "path",
			1 => "debug",
			_ => unreachable!(),
		}
	}

	fn tonemap_text(tonemap: usize) -> &'static str {
		match tonemap {
			0 => "aces",
			1 => "tony mcmapface",
			_ => unreachable!(),
		}
	}

	pub fn render(
		&mut self, device: &Device, ctx: &Context, stats: Option<CullStats>, pt: Option<(ExposureStats, u32)>,
	) {
		Window::new("debug").open(&mut self.enabled).show(ctx, |ui| {
			let mut sel = self.render_mode as usize;
			ComboBox::from_label("render mode")
				.selected_text(Self::mode_text(sel))
				.show_index(ui, &mut sel, 2, Self::mode_text);
			self.render_mode = match sel {
				0 => RenderMode::Path,
				1 => RenderMode::Debug,
				_ => unreachable!(),
			};

			match self.render_mode {
				RenderMode::Path => {
					let mut sel = self.tonemap as usize;
					ComboBox::from_label("tonemap")
						.selected_text(Self::tonemap_text(sel))
						.show_index(ui, &mut sel, 2, Self::tonemap_text);
					self.tonemap = match sel {
						0 => Tonemap::Aces,
						1 => Tonemap::TonyMcMapface,
						_ => unreachable!(),
					};
				},
				RenderMode::Debug => {
					let mut sel = self.debug_vis.to_u32() as usize;
					ComboBox::from_label("debug vis")
						.selected_text(Self::vis_text(sel))
						.show_index(ui, &mut sel, 11, Self::vis_text);
					self.debug_vis = match sel {
						0 => DebugVis::Triangles,
						1 => DebugVis::Meshlets,
						2 => DebugVis::Overdraw(self.scale),
						3 => DebugVis::HwSw,
						4 => DebugVis::Normals,
						5 => DebugVis::Uvs,
						6 => DebugVis::Error,
						7 => DebugVis::BaseColor,
						8 => DebugVis::Roughness,
						9 => DebugVis::Metallic,
						10 => DebugVis::Emissive,
						_ => unreachable!(),
					};

					match &mut self.debug_vis {
						DebugVis::Overdraw(s) => {
							ui.horizontal(|ui| {
								ui.add(DragValue::new(&mut self.scale).speed(0.01).range(0.0..=1.0));
							});
							*s = self.scale;
						},
						_ => {},
					}
				},
			}

			ui.horizontal(|ui| {
				ui.label("hotreload: ");
				match device.hotreload_status() {
					HotreloadStatus::Waiting => ui.label("ready"),
					HotreloadStatus::Recompiling => ui.spinner(),
					HotreloadStatus::Errored => ui.label("errored"),
				}
			});

			if let Some(stats) = stats {
				ui.label("early");
				Self::pass_stats(ui, stats.early);
				ui.label("late");
				Self::pass_stats(ui, stats.late);
			}

			if let Some((exp, samples)) = pt {
				ui.label(format!("samples: {}", samples));

				ui.label(format!("exposure: {:.2}", exp.exposure));

				Plot::new("exposure histogram")
					.allow_zoom(false)
					.allow_scroll(false)
					.allow_drag(false)
					.allow_boxed_zoom(false)
					.show_background(false)
					.show_grid(false)
					.show_x(false)
					.show_y(false)
					.x_axis_position(VPlacement::Bottom)
					.y_axis_position(HPlacement::Left)
					.auto_bounds([false, true].into())
					.include_x(0.0)
					.include_x(255.0)
					.x_axis_formatter(|_, _| "".to_string())
					.y_axis_formatter(|_, _| "".to_string())
					.show(ui, |ui| {
						ui.bar_chart(BarChart::new(
							exp.histogram
								.into_iter()
								.enumerate()
								.map(|(i, x)| Bar::new(i as _, x as _).width(1.0))
								.collect(),
						));

						ui.vline(VLine::new(
							(exp.exposure - ExposureCalc::MIN_EXPOSURE)
								/ (ExposureCalc::MAX_EXPOSURE - ExposureCalc::MIN_EXPOSURE)
								* 255.0,
						));
						ui.vline(VLine::new(
							(exp.target_exposure - ExposureCalc::MIN_EXPOSURE)
								/ (ExposureCalc::MAX_EXPOSURE - ExposureCalc::MIN_EXPOSURE)
								* 255.0,
						));
					});
			}
		});
	}

	fn pass_stats(ui: &mut Ui, pass: PassStats) {
		ui.label(format!("instances: {}", pass.instances));
		ui.label(format!("candidate meshlets: {}", pass.candidate_meshlets));
		ui.label(format!("hw meshlets: {}", pass.hw_meshlets));
		ui.label(format!("sw meshlets: {}", pass.sw_meshlets));
	}

	pub fn render_mode(&self) -> RenderMode { self.render_mode }

	pub fn tonemap(&self) -> Tonemap { self.tonemap }

	pub fn debug_vis(&self) -> DebugVis { self.debug_vis }
}
