use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{rref::RRef, scene, AssetSystem, Uuid};
use radiance_egui::to_texture_id;
use radiance_graph::{device::Device, graph::Frame, Result};
use radiance_passes::{
	debug::mesh::DebugMesh,
	mesh::{RenderInfo, VisBuffer},
};
use tracing::{event, Level};
use vek::Vec2;
use winit::event::WindowEvent;

use crate::{
	ui::{
		debug::Debug,
		render::{
			camera::{CameraController, Mode},
			picking::Picker,
		},
	},
	window::Window,
};

mod camera;
mod picking;

enum Scene {
	None,
	Unloaded(Uuid),
	Loaded(RRef<scene::Scene>),
}

pub struct Renderer {
	scene: Scene,
	visbuffer: VisBuffer,
	picker: Picker,
	debug: DebugMesh,
	camera: CameraController,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			scene: Scene::None,
			visbuffer: VisBuffer::new(device)?,
			picker: Picker::new(device)?,
			debug: DebugMesh::new(device)?,
			camera: CameraController::new(),
		})
	}

	pub fn set_scene(&mut self, scene: Uuid) { self.scene = Scene::Unloaded(scene); }

	pub fn render<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, window: &Window, debug: &Debug,
		system: Option<&AssetSystem>,
	) {
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

	fn render_inner<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, ui: &mut Ui, window: &Window,
		system: Option<&AssetSystem>, debug: &Debug,
	) -> Option<bool> {
		let Some(system) = system else {
			return Some(true);
		};
		let scene = match self.scene {
			Scene::None => return Some(true),
			Scene::Unloaded(s) => match system.initialize::<scene::Scene>(frame.device(), s) {
				Ok(s) => {
					self.scene = Scene::Loaded(s.clone());
					s
				},
				Err(e) => {
					event!(Level::ERROR, "error loading scene: {:?}", e);
					return None;
				},
			},
			Scene::Loaded(ref s) => s.clone(),
		};

		let rect = ui.available_rect_before_wrap();
		let size = rect.size();

		let pointer_pos = ctx.input(|x| x.pointer.hover_pos());
		let pointer_in = pointer_pos.map(|x| rect.contains(x)).unwrap_or(false);
		if ctx.input(|x| pointer_in && x.pointer.button_down(PointerButton::Secondary)) {
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

		let clicked = ctx.input(|x| pointer_in && x.pointer.primary_clicked());
		let selected = self.picker.run(
			frame,
			visbuffer.reader,
			clicked.then(|| pointer_pos.map(|x| x - rect.min)).flatten(),
		);

		let img = self
			.debug
			.run(frame, debug.debug_vis(), visbuffer, selected.into_iter());
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

	pub unsafe fn destroy(mut self, device: &Device) {
		self.scene = Scene::None;
		self.visbuffer.destroy(device);
		self.picker.destroy(device);
		self.debug.destroy(device);
	}
}
