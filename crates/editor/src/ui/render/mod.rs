use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_asset_runtime::{rref::RRef, scene, AssetRuntime};
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::to_texture_id;
use radiance_graph::Result;
use radiance_passes::{
	debug::meshlet::DebugMeshlets,
	mesh::visbuffer::{RenderInfo, VisBuffer},
};
use tracing::{event, Level};
use vek::Vec2;
use winit::event::WindowEvent;

use crate::{
	ui::render::{
		camera::{CameraController, Mode},
		debug::DebugWindows,
	},
	window::Window,
};

mod camera;
mod debug;

enum Scene {
	None,
	Unloaded(Uuid),
	Loaded(RRef<scene::Scene>),
}

pub struct Renderer {
	scene: Scene,
	visbuffer: VisBuffer,
	debug: DebugMeshlets,
	runtime: AssetRuntime,
	debug_windows: DebugWindows,
	camera: CameraController,
}

impl Renderer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		Ok(Self {
			scene: Scene::None,
			visbuffer: VisBuffer::new(device, core)?,
			debug: DebugMeshlets::new(device, core)?,
			runtime: AssetRuntime::new(device)?,
			debug_windows: DebugWindows::new(),
			camera: CameraController::new(),
		})
	}

	pub fn set_scene(&mut self, core: &mut RenderCore, scene: Uuid) { self.scene = Scene::Unloaded(scene); }

	pub fn render<'pass, S: AssetSource>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, window: &Window,
		system: Option<&AssetSystem<S>>,
	) {
		self.runtime.tick(frame.ctx());
		CentralPanel::default().show(ctx, |ui| {
			if let Some(x) = self.render_inner(device, frame, ctx, ui, window, system) {
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
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, ui: &mut Ui,
		window: &Window, system: Option<&AssetSystem<S>>,
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

		let visbuffer = self.visbuffer.run(
			device,
			frame,
			RenderInfo {
				scene,
				camera: self.camera.get(),
				cull_camera: self.debug_windows.cull_camera(),
				size: Vec2::new(size.x as u32, size.y as u32),
			},
		);
		let debug = self.debug.run(frame, visbuffer);
		ui.image((to_texture_id(debug), size));

		Some(false)
	}

	pub fn draw_debug_menu(&mut self, ui: &mut Ui) { self.debug_windows.draw_menu(ui) }

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

	pub fn draw_debug_windows(&mut self, ctx: &Context, device: &CoreDevice) {
		self.debug_windows.draw(ctx, device, &self.camera);
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		self.camera.on_window_event(window, event);
	}

	pub unsafe fn destroy(mut self, device: &CoreDevice) {
		self.scene = Scene::None;
		self.visbuffer.destroy(device);
		self.debug.destroy(device);
		self.runtime.destroy(device);
	}
}

