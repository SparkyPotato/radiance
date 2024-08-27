use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_egui::to_texture_id;
use radiance_graph::{
	device::{Device, QueueSyncs},
	graph::Frame,
	Result,
};
use radiance_passes::{
	asset::{rref::RRef, scene, AssetRuntime},
	debug::mesh::DebugMesh,
	mesh::{RenderInfo, VisBuffer},
	tonemap::aces::AcesTonemap,
};
use tracing::{event, Level};
use vek::Vec2;
use winit::event::WindowEvent;

use crate::{
	ui::{
		debug::Debug,
		render::camera::{CameraController, Mode},
	},
	window::Window,
};

mod camera;

enum Scene {
	None,
	Unloaded(Uuid),
	Loaded((RRef<scene::Scene>, QueueSyncs)),
}

pub struct Renderer {
	scene: Scene,
	visbuffer: VisBuffer,
	debug: DebugMesh,
	// cpu_path: CpuPath,
	tonemap: AcesTonemap,
	runtime: AssetRuntime,
	camera: CameraController,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			scene: Scene::None,
			visbuffer: VisBuffer::new(device)?,
			debug: DebugMesh::new(device)?,
			// cpu_path: CpuPath::new(),
			tonemap: AcesTonemap::new(device)?,
			runtime: AssetRuntime::new(device)?,
			camera: CameraController::new(),
		})
	}

	pub fn set_scene(&mut self, scene: Uuid) { self.scene = Scene::Unloaded(scene); }

	pub fn render<'pass, S: AssetSource>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, window: &Window, debug: &Debug,
		system: Option<&AssetSystem<S>>,
	) {
		self.runtime.tick(frame);
		CentralPanel::default().show(ctx, |ui| {
			if let Some(x) = self.render_inner(frame, ctx, ui, window, system, debug) {
				if x {
					ui.centered_and_justified(|ui| {
						ui.label(RichText::new("no scene loaded").size(20.0));
					});
				}
			} else {
				ui.centered_and_justified(|ui| {
					ui.label(RichText::new("error rendering scene").size(20.0));
				});
			}
		});
	}

	fn render_inner<'pass, S: AssetSource>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, ui: &mut Ui, window: &Window,
		system: Option<&AssetSystem<S>>, debug: &Debug,
	) -> Option<bool> {
		let Some(system) = system else {
			return Some(true);
		};
		let (scene, wait) = match self.scene {
			Scene::None => return Some(true),
			Scene::Unloaded(s) => {
				match self
					.runtime
					.load(frame.device(), system, frame.async_exec().unwrap(), |l| l.load_scene(s))
				{
					Ok(s) => {
						self.scene = Scene::Loaded(s.clone());
						s
					},
					Err(e) => {
						event!(Level::ERROR, "error loading scene: {:?}", e);
						return None;
					},
				}
			},
			Scene::Loaded(ref s) => s.clone(),
		};

		let rect = ui.available_rect_before_wrap();
		let size = rect.size();

		if ctx.input(|x| {
			let p = &x.pointer;
			p.hover_pos().map(|x| rect.contains(x)).unwrap_or(false) && p.button_down(PointerButton::Secondary)
		}) {
			self.camera.set_mode(window, Mode::Camera);
		} else {
			self.camera.set_mode(window, Mode::Default);
		}
		self.camera.control(ctx);

		let s = Vec2::new(size.x as u32, size.y as u32);
		let visbuffer = self.visbuffer.run(
			frame,
			RenderInfo {
				scene,
				camera: self.camera.get(),
				size: s,
				debug_info: debug.debug_vis().requires_debug_info(),
			},
		);

		let img = self.debug.run(frame, debug.debug_vis(), visbuffer);
		ui.image((to_texture_id(img), size));

		Some(false)
	}

	pub fn draw_camera_menu(&mut self, ui: &mut Ui) {
		match self.scene {
			Scene::Loaded(ref scene) => {
				for c in scene.0.cameras.iter() {
					if ui.button(&c.name).clicked() {
						self.camera.set(c);
					}
				}
			},
			_ => {},
		}
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		self.camera.on_window_event(window, event);
	}

	pub unsafe fn destroy(mut self, device: &Device) {
		self.scene = Scene::None;
		self.visbuffer.destroy(device);
		self.debug.destroy(device);
		self.runtime.destroy(device);
		self.tonemap.destroy(device);
	}
}
