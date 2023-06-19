use egui::{CentralPanel, Context, Key, PointerButton, RichText};
use radiance_asset::{AssetSource, AssetSystem, Uuid};
use radiance_asset_runtime::AssetRuntime;
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_egui::to_texture_id;
use radiance_graph::Result;
use radiance_passes::{
	debug::meshlet::DebugMeshlets,
	mesh::{
		cull::{Camera, Cull},
		visbuffer::VisBuffer,
	},
};
use vek::{num_traits::FloatConst, Mat4, Vec2, Vec3, Vec4};

use crate::window::Window;

#[derive(Copy, Clone, PartialEq)]
enum Mode {
	Camera,
	Default,
}

pub struct Renderer {
	scene: Option<Uuid>,
	cull: Cull,
	visbuffer: VisBuffer,
	debug: DebugMeshlets,
	runtime: AssetRuntime,
	pos: Vec3<f32>,
	pitch: f32,
	yaw: f32,
	move_speed: f32,
	mode: Mode,
}

impl Renderer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		Ok(Self {
			scene: None,
			cull: Cull::new(device, core)?,
			visbuffer: VisBuffer::new(device, core)?,
			debug: DebugMeshlets::new(device, core)?,
			runtime: AssetRuntime::new(),
			pos: Vec3::zero(),
			pitch: 0.0,
			yaw: 0.0,
			move_speed: 1.0,
			mode: Mode::Default,
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
		system: Option<&mut AssetSystem<S>>,
	) {
		match self.mode {
			Mode::Camera => ctx.input(|x| {
				let dt = x.unstable_dt;
				let speed = self.move_speed * dt;

				let delta = x.pointer.delta();
				let delta = Vec2::new(delta.x, delta.y) * dt * 0.5;
				self.pitch += delta.y;
				self.yaw += delta.x;
				self.pitch = self.pitch.clamp(-f32::FRAC_PI_2(), f32::FRAC_PI_2());

				let yaw = Mat4::rotation_3d(-self.yaw, Vec3::unit_y());
				let forward = (yaw * Vec4::unit_z()).xyz();
				let right = (yaw * Vec4::unit_x()).xyz();

				for key in x.keys_down.iter() {
					self.pos += match key {
						Key::W => forward * speed,
						Key::S => forward * -speed,
						Key::A => right * speed,
						Key::D => right * -speed,
						Key::Q => Vec3::new(0.0, speed, 0.0),
						Key::E => Vec3::new(0.0, -speed, 0.0),
						_ => continue,
					};
				}

				let factor = 2f32.powf(x.scroll_delta.y / 50.0);
				self.move_speed *= factor;
			}),
			Mode::Default => {},
		}

		CentralPanel::default().show(ctx, |ui| {
			let inner = || {
				let Some(scene) = self.scene else { return true };
				let Some(system) = system else { return true };

				let rect = ui.available_rect_before_wrap();
				let size = rect.size();
				let aspect = size.x / size.y;

				if ctx.input(|x| {
					let p = &x.pointer;
					p.hover_pos().map(|x| rect.contains(x)).unwrap_or(false) && p.button_down(PointerButton::Secondary)
				}) {
					self.set_mode(window, Mode::Camera);
				} else {
					self.set_mode(window, Mode::Default);
				}

				if let Some(ticket) = self.runtime.load_scene(device, frame.ctx(), scene, system).unwrap() {
					let mut pass = frame.pass("init");
					pass.wait_on(ticket.as_info());
					pass.build(|_| {});
				}
				let scene = self.runtime.get_scene(scene).unwrap();

				let cull = self.cull.run(
					frame,
					scene,
					Camera {
						view: Mat4::identity().rotated_y(self.yaw).rotated_x(self.pitch)
							* Mat4::translation_3d(self.pos),
						proj: infinite_projection(aspect, 90f32.to_radians(), 0.01),
					},
				);
				let visbuffer = self
					.visbuffer
					.run(frame, scene, cull, Vec2::new(size.x as u32, size.y as u32));
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

	fn set_mode(&mut self, window: &Window, mode: Mode) {
		if self.mode != mode {
			match mode {
				Mode::Camera => {
					window.window.set_cursor_visible(false);
				},
				Mode::Default => {
					window.window.set_cursor_visible(true);
				},
			}
			self.mode = mode;
		}
		match mode {
			Mode::Camera => {
				// let size = window.window.inner_size();
				// window
				// 	.window
				// 	.set_cursor_position(PhysicalPosition::new(size.width / 2, size.height / 2))
				// 	.unwrap();
			},
			_ => {},
		}
	}
}

fn infinite_projection(aspect: f32, fov: f32, near: f32) -> Mat4<f32> {
	let h = 1.0 / (fov / 2.0).tan();
	let w = h * (1.0 / aspect);
	Mat4::new(
		w, 0.0, 0.0, 0.0, //
		0.0, -h, 0.0, 0.0, //
		0.0, 0.0, 0.0, near, //
		0.0, 0.0, -1.0, 0.0, //
	)
}
