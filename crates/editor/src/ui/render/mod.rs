use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_asset_runtime::{rref::RRef, scene, AssetRuntime};
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::to_texture_id;
use radiance_graph::Result;
use radiance_passes::{
	cpu_path::{self, CpuPath},
	debug::meshlet::DebugMeshlets,
	ground_truth::{self, GroundTruth},
	mesh::visbuffer::{RenderInfo, VisBuffer},
	tonemap::aces::AcesTonemap,
};
use tracing::{event, Level};
use vek::Vec2;
use winit::event::WindowEvent;

use crate::{
	ui::{
		debug::{Debug, RenderMode},
		render::camera::{CameraController, Mode},
	},
	window::Window,
};

mod camera;

enum Scene {
	None,
	Unloaded(Uuid),
	Loaded(RRef<scene::Scene>),
}

pub struct Renderer {
	scene: Scene,
	visbuffer: VisBuffer,
	debug: DebugMeshlets,
	ground_truth: GroundTruth,
	cpu_path: CpuPath,
	tonemap: AcesTonemap,
	runtime: AssetRuntime,
	camera: CameraController,
}

impl Renderer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		Ok(Self {
			scene: Scene::None,
			visbuffer: VisBuffer::new(device, core)?,
			debug: DebugMeshlets::new(device, core)?,
			ground_truth: GroundTruth::new(device, core)?,
			cpu_path: CpuPath::new(),
			tonemap: AcesTonemap::new(device, core)?,
			runtime: AssetRuntime::new(device)?,
			camera: CameraController::new(),
		})
	}

	pub fn set_scene(&mut self, scene: Uuid) { self.scene = Scene::Unloaded(scene); }

	pub fn render<'pass, S: AssetSource>(
		&'pass mut self, device: &'pass CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, window: &Window,
		debug: &Debug, system: Option<&AssetSystem<S>>,
	) {
		self.runtime.tick(frame.ctx());
		CentralPanel::default().show(ctx, |ui| {
			if let Some(x) = self.render_inner(device, frame, ctx, ui, window, system, debug) {
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
		&'pass mut self, device: &'pass CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, ui: &mut Ui,
		window: &Window, system: Option<&AssetSystem<S>>, debug: &Debug,
	) -> Option<bool> {
		let Some(system) = system else {
			return Some(true);
		};
		let (scene, ticket) = match self.scene {
			Scene::None => return Some(true),
			Scene::Unloaded(s) => match self
				.runtime
				.load(device, frame.ctx(), system, |r, l| r.load_scene(l, s))
			{
				Ok((s, t)) => {
					self.scene = Scene::Loaded(s.clone());
					(s, Some(t))
				},
				Err(e) => {
					event!(Level::ERROR, "error loading scene: {:?}", e);
					return None;
				},
			},
			Scene::Loaded(ref s) => (s.clone(), None),
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

		if let Some(ticket) = ticket {
			let mut pass = frame.pass("wait for staging");
			pass.wait_on(ticket.as_info());
			pass.build(|_| {});
		}

		let s = Vec2::new(size.x as u32, size.y as u32);
		let img = match debug.render_mode() {
			RenderMode::Realtime => {
				let visbuffer = self.visbuffer.run(
					device,
					frame,
					RenderInfo {
						scene,
						camera: self.camera.get(),
						cull_camera: debug.cull_camera(),
						size: s,
					},
				);
				self.debug.run(frame, visbuffer)
			},
			RenderMode::GroundTruth => {
				let rt = self.ground_truth.run(
					device,
					frame,
					ground_truth::RenderInfo {
						scene,
						materials: self.runtime.materials(),
						camera: self.camera.get(),
						size: s,
					},
				);
				self.tonemap.run(frame, rt)
			},
			RenderMode::CpuPath => {
				let rt = self.cpu_path.run(
					device,
					frame,
					cpu_path::RenderInfo {
						scene,
						camera: self.camera.get(),
						size: s,
					},
				);
				self.tonemap.run(frame, rt)
			},
		};
		ui.image((to_texture_id(img), size));

		Some(false)
	}

	pub fn draw_camera_menu(&mut self, ui: &mut Ui) {
		match self.scene {
			Scene::Loaded(ref scene) => {
				for c in scene.cameras.iter() {
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

	pub unsafe fn destroy(mut self, device: &CoreDevice) {
		self.scene = Scene::None;
		self.visbuffer.destroy(device);
		self.debug.destroy(device);
		self.ground_truth.destroy(device);
		drop(self.cpu_path);
		self.runtime.destroy(device);
		self.tonemap.destroy(device);
	}
}
