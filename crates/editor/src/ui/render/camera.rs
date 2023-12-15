use egui::{Context, Key};
use radiance_asset::scene;
use radiance_passes::mesh::visbuffer::Camera;
use vek::{num_traits::FloatConst, Mat4, Vec2, Vec3, Vec4};

use crate::window::Window;

#[derive(Copy, Clone, PartialEq)]
pub enum Mode {
	Camera,
	Default,
}

pub struct CameraController {
	pos: Vec3<f32>,
	pitch: f32,
	yaw: f32,
	move_speed: f32,
	mode: Mode,
}

impl CameraController {
	pub fn new() -> Self {
		Self {
			pos: Vec3::zero(),
			pitch: 0.0,
			yaw: 0.0,
			move_speed: 1.0,
			mode: Mode::Default,
		}
	}

	pub fn set_mode(&mut self, window: &Window, mode: Mode) {
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

	pub fn control(&mut self, ctx: &Context) {
		match self.mode {
			Mode::Camera => ctx.input(|x| {
				let dt = x.stable_dt;
				let speed = self.move_speed * dt;

				let delta = x.pointer.delta();
				let delta = Vec2::new(delta.x, delta.y) * 0.005;
				self.pitch += delta.y;
				self.yaw += delta.x;
				self.pitch = self.pitch.clamp(-f32::FRAC_PI_2(), f32::FRAC_PI_2());

				let yaw = Mat4::rotation_3d(self.yaw, Vec3::unit_y());
				let forward = (yaw * Vec4::unit_z()).xyz();
				let right = (yaw * Vec4::unit_x()).xyz();

				let mut offset = Vec3::zero();
				for key in x.keys_down.iter() {
					offset += match key {
						Key::W => forward,
						Key::S => -forward,
						Key::D => right,
						Key::A => -right,
						Key::E => Vec3::unit_y(),
						Key::Q => -Vec3::unit_y(),
						_ => continue,
					};
				}
				if offset != Vec3::zero() {
					offset.normalize();
					self.pos += offset * speed;
				}

				let factor = 2f32.powf(x.scroll_delta.y / 50.0);
				self.move_speed *= factor;
			}),
			Mode::Default => {},
		}
	}

	pub fn set(&mut self, camera: &scene::Camera) {
		let view = camera.view;
		self.pos = -view.cols.w.xyz();
		self.pitch = -(-view.cols.x.z).atan2((view.cols.x.x * view.cols.x.x + view.cols.x.y * view.cols.x.y).sqrt());
		self.yaw = -(view.cols.x.y / self.pitch.cos()).atan2(view.cols.x.x / self.pitch.cos());
	}

	pub fn get(&self) -> Camera {
		Camera {
			view: Mat4::identity().rotated_y(-self.yaw).rotated_x(-self.pitch) * Mat4::translation_3d(-self.pos),
			fov: 90f32.to_radians(),
			near: 0.01,
		}
	}
}
