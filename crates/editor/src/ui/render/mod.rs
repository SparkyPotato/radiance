use egui::{CentralPanel, Context, PointerButton, RichText, Ui};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_asset_runtime::AssetRuntime;
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::to_texture_id;
use radiance_graph::Result;
use radiance_passes::{debug::meshlet::DebugMeshlets, mesh::visbuffer::VisBuffer};
use vek::Vec2;

use crate::{
	ui::render::{
		camera::{CameraController, Mode},
		debug::DebugWindows,
	},
	window::Window,
};

mod camera;
mod debug;

pub struct Renderer {
	scene: Option<Uuid>,
	visbuffer: VisBuffer,
	debug: DebugMeshlets,
	runtime: AssetRuntime,
	camera: CameraController,
	debug_windows: DebugWindows,
}

impl Renderer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		Ok(Self {
			scene: None,
			visbuffer: VisBuffer::new(device, core)?,
			debug: DebugMeshlets::new(device, core)?,
			runtime: AssetRuntime::new(),
			camera: CameraController::new(),
			debug_windows: DebugWindows::new(),
		})
	}

	pub fn set_scene(&mut self, core: &mut RenderCore, scene: Uuid) {
		if let Some(scene) = self.scene {
			self.runtime.unload_scene(core, scene);
		}
		self.scene = Some(scene);
	}

	pub fn render<'pass, S: AssetSource>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, window: &Window,
		system: Option<&AssetSystem<S>>,
	) {
		CentralPanel::default().show(ctx, |ui| {
			if self.render_inner(device, frame, ctx, ui, window, system) {
				ui.centered_and_justified(|ui| {
					ui.label(RichText::new("no scene loaded").size(20.0));
				});
			}
		});
	}

	fn render_inner<'pass, S: AssetSource>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, ui: &mut Ui,
		window: &Window, system: Option<&AssetSystem<S>>,
	) -> bool {
		let Some(scene) = self.scene else { return true };
		let Some(system) = system else { return true };

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

		let (_, ticket) = self.runtime.load_scene(device, frame.ctx(), scene, system).unwrap();
		if let Some(ticket) = ticket {
			let mut pass = frame.pass("init");
			pass.wait_on(ticket.as_info());
			pass.build(|_| {});
		}

		let scene = self.runtime.get_scene(scene).unwrap();
		let visbuffer = self.visbuffer.run(
			frame,
			&scene,
			self.camera.get(),
			self.debug_windows.cull_camera(),
			Vec2::new(size.x as u32, size.y as u32),
		);
		let debug = self.debug.run(frame, visbuffer);
		ui.image((to_texture_id(debug), size));

		false
	}

	pub fn draw_debug_menu(&mut self, ui: &mut Ui) { self.debug_windows.draw_menu(ui) }

	pub fn draw_debug_windows(&mut self, ctx: &Context) { self.debug_windows.draw(ctx, &self.camera); }

	pub unsafe fn destroy(self, device: &CoreDevice) {
		self.visbuffer.destroy(device);
		self.debug.destroy(device);
		self.runtime.destroy(device);
	}
}
