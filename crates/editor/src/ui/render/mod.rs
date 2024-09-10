use std::sync::Arc;

use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{rref::RRef, scene, AssetSystem, Uuid};
use radiance_egui::to_texture_id;
use radiance_graph::{device::Device, graph::Frame, Result};
use radiance_passes::{
	debug::mesh::DebugMesh,
	mesh::{RenderInfo, VisBuffer},
};
use vek::Vec2;
use winit::event::WindowEvent;

use crate::{
	ui::{
		debug::Debug,
		notif::NotifStack,
		render::{
			camera::{CameraController, Mode},
			edit::Editor,
			picking::Picker,
		},
		task::{TaskNotif, TaskPool},
	},
	window::Window,
};

mod camera;
mod edit;
mod picking;

enum Scene {
	None,
	Unloaded(Uuid),
	Loading(oneshot::Receiver<Option<RRef<scene::Scene>>>),
	Loaded(RRef<scene::Scene>),
}

pub struct Renderer {
	scene: Scene,
	editor: Editor,
	visbuffer: VisBuffer,
	picker: Picker,
	debug: DebugMesh,
	camera: CameraController,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			scene: Scene::None,
			editor: Editor::new(),
			visbuffer: VisBuffer::new(device)?,
			picker: Picker::new(device)?,
			debug: DebugMesh::new(device)?,
			camera: CameraController::new(),
		})
	}

	pub fn set_scene(&mut self, scene: Uuid) { self.scene = Scene::Unloaded(scene); }

	pub fn render<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, window: &Window, debug: &Debug,
		system: Option<Arc<AssetSystem>>, notifs: &mut NotifStack, pool: &TaskPool,
	) {
		CentralPanel::default().show(ctx, |ui| {
			if let Some(x) = self.render_inner(frame, ctx, ui, window, debug, system, notifs, pool) {
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
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, ui: &mut Ui, window: &Window, debug: &Debug,
		system: Option<Arc<AssetSystem>>, notifs: &mut NotifStack, pool: &TaskPool,
	) -> Option<bool> {
		let Some(system) = system else {
			return Some(true);
		};
		let scene = match self.scene {
			Scene::None => return Some(true),
			Scene::Unloaded(sc) => {
				let dev = frame.device().clone();
				let (s, r) = oneshot::channel();
				notifs.push(
					"loading scene",
					TaskNotif::new(
						"loading",
						pool.spawn(move || match system.initialize::<scene::Scene>(&dev, sc) {
							Ok(sc) => {
								s.send(Some(sc)).unwrap();
								Ok(())
							},
							Err(e) => {
								s.send(None).unwrap();
								Err(e)
							},
						}),
					),
				);
				self.scene = Scene::Loading(r);
				return Some(true);
			},
			Scene::Loading(ref mut r) => match r.try_recv() {
				Ok(Some(s)) => {
					let s = s.clone();
					self.scene = Scene::Loaded(s.clone());
					s
				},
				Ok(None) => {
					self.scene = Scene::None;
					return Some(true);
				},
				Err(_) => return Some(true),
			},
			Scene::Loaded(ref s) => s.clone(),
		};

		self.editor.render(ui, &scene, &mut self.picker);

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

	pub fn undo(&mut self, notifs: &mut NotifStack) {
		match self.scene {
			Scene::Loaded(ref scene) => {
				self.editor.undo(scene, notifs);
			},
			_ => {},
		}
	}

	pub fn save(&mut self, system: Option<Arc<AssetSystem>>, notifs: &mut NotifStack, pool: &TaskPool) {
		if let Scene::Loaded(ref scene) = self.scene {
			self.editor.save(system, scene, notifs, pool);
		}
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
