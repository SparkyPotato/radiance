use egui::{CentralPanel, Context, RichText};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_asset_runtime::AssetRuntime;
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::to_texture_id;
use radiance_graph::Result;
use radiance_passes::{
	debug::meshlet::DebugMeshlets,
	mesh::{cull::Cull, visbuffer::VisBuffer},
};
use vek::{Mat4, Vec2, Vec3};

pub struct Renderer {
	scene: Option<Uuid>,
	cull: Cull,
	visbuffer: VisBuffer,
	debug: DebugMeshlets,
	runtime: AssetRuntime,
}

impl Renderer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		Ok(Self {
			scene: None,
			cull: Cull::new(device, core)?,
			visbuffer: VisBuffer::new(device, core)?,
			debug: DebugMeshlets::new(device, core)?,
			runtime: AssetRuntime::new(),
		})
	}

	pub fn set_scene(&mut self, core: &mut RenderCore, scene: Uuid) {
		if let Some(scene) = self.scene {
			self.runtime.unload_scene(core, scene);
		}
		self.scene = Some(scene);
	}

	pub fn render<'pass, S: AssetSource>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context,
		system: Option<&mut AssetSystem<S>>,
	) {
		CentralPanel::default().show(ctx, |ui| {
			let inner = || {
				let Some(scene) = self.scene else { return true };
				let Some(system) = system else { return true };

				if let Some(ticket) = self.runtime.load_scene(device, frame.ctx(), scene, system).unwrap() {
					let mut pass = frame.pass("init");
					pass.wait_on(ticket.as_info());
					pass.build(|_| {});
				}
				let scene = self.runtime.get_scene(scene).unwrap();

				let cull = self.cull.run(frame, scene);
				let size = ui.available_size_before_wrap();

				let aspect = size.x / size.y;
				let rad = 90f32.to_radians();
				let h = 1.0 / (rad / 2.0).tan();
				let w = h * (1.0 / aspect);
				let proj = Mat4::new(
					w, 0.0, 0.0, 0.0, //
					0.0, -h, 0.0, 0.0, //
					0.0, 0.0, 0.0, 0.1, //
					0.0, 0.0, -1.0, 0.0, //
				);
				let view = Mat4::look_at_rh(Vec3::broadcast(10.0), Vec3::broadcast(0.0), Vec3::unit_y());
				let visbuffer =
					self.visbuffer
						.run(frame, scene, cull, Vec2::new(size.x as u32, size.y as u32), proj * view);
				let debug = self.debug.run(frame, visbuffer);

				ui.image(to_texture_id(debug), size);
				false
			};

			if inner() {
				ui.centered_and_justified(|ui| {
					ui.label(RichText::new("no scene loaded").size(20.0));
				});
			}
		});
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		self.cull.destroy(device);
		self.visbuffer.destroy(device);
		self.debug.destroy(device);
		self.runtime.destroy(device);
	}
}