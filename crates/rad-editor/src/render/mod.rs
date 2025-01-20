use rad_core::Engine;
use rad_graph::{graph::Frame, Result};
use rad_renderer::{
	debug::mesh::DebugMesh,
	mesh::{self, VisBuffer},
	pt::{self, PathTracer},
	scene::{camera::CameraSceneInfo, WorldRenderer},
	sky::SkyLuts,
	tonemap::{
		aces::AcesTonemap,
		agx::{AgXLook, AgXTonemap},
		exposure::ExposureCalc,
		tony_mc_mapface::TonyMcMapfaceTonemap,
	},
	vek::Vec2,
};
use rad_ui::{
	egui::{CentralPanel, Context, Image, PointerButton, Sense},
	to_texture_id,
};
use rad_window::winit::{event::WindowEvent, window::Window};

use crate::{
	render::{
		camera::{CameraController, Mode},
		debug::{DebugWindow, RenderMode, Tonemap},
	},
	world::WorldContext,
};

mod camera;
mod debug;

pub struct Renderer {
	pub debug_window: DebugWindow,
	sky: SkyLuts,
	visbuffer: VisBuffer,
	pt: PathTracer,
	exposure: ExposureCalc,
	aces: AcesTonemap,
	agx: AgXTonemap,
	tony_mcmapface: TonyMcMapfaceTonemap,
	debug: DebugMesh,
	camera: CameraController,
	frame: u64,
}

impl Renderer {
	pub fn new() -> Result<Self> {
		let device = Engine::get().global();
		Ok(Self {
			debug_window: DebugWindow::new(),
			sky: SkyLuts::new(device)?,
			visbuffer: VisBuffer::new(device)?,
			pt: PathTracer::new(device)?,
			exposure: ExposureCalc::new(device)?,
			aces: AcesTonemap::new(device)?,
			agx: AgXTonemap::new(device)?,
			tony_mcmapface: TonyMcMapfaceTonemap::new(device)?,
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
		let (stats, pt) = CentralPanel::default()
			.show(ctx, |ui| {
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
				let mut rend = WorldRenderer::new(world.world_mut(), frame.arena());

				rend.set_input(CameraSceneInfo {
					aspect: size.x / size.y,
				});

				let vis = self.debug_window.debug_vis();
				let (img, stats, exp) = match self.debug_window.render_mode() {
					RenderMode::Path => {
						let sky = self.sky.run(frame, &mut rend);
						let (hdr, s) = self.pt.run(
							frame,
							&mut rend,
							pt::RenderInfo {
								sky,
								size: Vec2::new(size.x as u32, size.y as u32),
							},
						);
						let exp = self.exposure.run(
							frame,
							hdr,
							self.debug_window.exposure_compensation(),
							ui.input(|x| x.stable_dt),
						);
						let img = match self.debug_window.tonemap() {
							Tonemap::Aces => self.aces.run(frame, hdr, exp.exposure),
							Tonemap::AgX => self.agx.run(frame, hdr, exp.exposure, AgXLook::default()),
							Tonemap::AgXPunchy => self.agx.run(frame, hdr, exp.exposure, AgXLook::punchy()),
							Tonemap::AgXFilmic => self.agx.run(frame, hdr, exp.exposure, AgXLook::filmic()),
							Tonemap::TonyMcMapface => self.tony_mcmapface.run(frame, hdr, exp.exposure),
						};
						(img, None, Some((exp, s)))
					},
					RenderMode::Debug => {
						let visbuffer = self.visbuffer.run(
							frame,
							&mut rend,
							mesh::RenderInfo {
								size: Vec2::new(size.x as u32, size.y as u32),
								debug_info: vis.requires_debug_info(),
							},
						);
						let img = self.debug.run(frame, vis, visbuffer, [].into_iter());
						(img, Some(visbuffer.stats), None)
					},
				};
				ui.put(rect, Image::new((to_texture_id(img), size)));

				(stats, exp)
			})
			.inner;

		self.debug_window.render(frame.device(), ctx, stats, pt);

		self.frame += 1;
	}

	pub unsafe fn destroy(self) {
		self.sky.destroy();
		self.visbuffer.destroy();
		self.pt.destroy();
		self.exposure.destroy();
		self.aces.destroy();
		self.tony_mcmapface.destroy();
		self.debug.destroy();
	}
}
