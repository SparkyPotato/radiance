use egui::Context;
use radiance_asset::scene;
use radiance_passes::mesh::Camera;
use vek::{num_traits::FloatConst, Mat4, Vec2, Vec3, Vec4};
use winit::{
	dpi::PhysicalPosition,
	event::{ElementState, MouseScrollDelta, WindowEvent},
	keyboard::{KeyCode, PhysicalKey},
	window::CursorGrabMode,
};

use crate::window::Window;

#[derive(Default)]
struct MouseGrabber {
	last_pos: PhysicalPosition<f64>,
	manual_lock: bool,
}

impl MouseGrabber {
	fn cursor_moved(&mut self, window: &Window, pos: PhysicalPosition<f64>) {
		if self.manual_lock {
			window.window.set_cursor_position(self.last_pos).unwrap();
		} else {
			self.last_pos = pos;
		}
	}

	fn grab(&mut self, window: &Window, grab: bool) {
		if grab {
			if window.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
				window.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
				self.manual_lock = true;
			}
		} else {
			self.manual_lock = false;
			window.window.set_cursor_grab(CursorGrabMode::None).unwrap();
		}
		window.window.set_cursor_visible(!grab);
	}
}

#[derive(Copy, Clone, PartialEq)]
pub enum Mode {
	Camera,
	Default,
}

pub struct CameraController {
	pos: Vec3<f32>,
	states: [bool; 6],
	pitch: f32,
	yaw: f32,
	move_speed: f32,
	mode: Mode,
	grabber: MouseGrabber,
}

impl CameraController {
	pub fn new() -> Self {
		Self {
			pos: Vec3::zero(),
			states: [false; 6],
			pitch: 0.0,
			yaw: 0.0,
			move_speed: 1.0,
			mode: Mode::Default,
			grabber: MouseGrabber::default(),
		}
	}

	pub fn set_mode(&mut self, window: &Window, mode: Mode) {
		if self.mode != mode {
			self.grabber.grab(window, mode == Mode::Camera);
			self.mode = mode;
		}
	}

	pub fn control(&mut self, ctx: &Context) {
		if self.mode != Mode::Camera {
			return;
		}

		let yaw = Mat4::rotation_3d(self.yaw, Vec3::unit_y());
		let forward = (yaw * Vec4::unit_z()).xyz();
		let right = (yaw * Vec4::unit_x()).xyz();
		let states = [forward, -forward, right, -right, Vec3::unit_y(), -Vec3::unit_y()];
		let dir = states
			.iter()
			.zip(self.states.iter())
			.filter_map(|(&dir, &state)| if state { Some(dir) } else { None })
			.fold(Vec3::zero(), |a, b| a + b);
		let dir = if dir == Vec3::zero() {
			return;
		} else {
			dir.normalized()
		};
		self.pos += dir * self.move_speed * ctx.input(|x| x.stable_dt);
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		match event {
			WindowEvent::CursorMoved { position, .. } => {
				self.grabber.cursor_moved(window, *position);
				let delta = Vec2::new(position.x as f32, position.y as _)
					- Vec2::new(self.grabber.last_pos.x as _, self.grabber.last_pos.y as _);
				let delta = Vec2::new(delta.x, delta.y) * 0.005;
				self.pitch += delta.y;
				self.yaw += delta.x;
				self.pitch = self.pitch.clamp(-f32::FRAC_PI_2(), f32::FRAC_PI_2());
			},
			WindowEvent::KeyboardInput {
				event,
				is_synthetic: false,
				..
			} => {
				let offset = match event.physical_key {
					PhysicalKey::Code(KeyCode::KeyW) => 0,
					PhysicalKey::Code(KeyCode::KeyS) => 1,
					PhysicalKey::Code(KeyCode::KeyD) => 2,
					PhysicalKey::Code(KeyCode::KeyA) => 3,
					PhysicalKey::Code(KeyCode::KeyE) => 4,
					PhysicalKey::Code(KeyCode::KeyQ) => 5,
					_ => return,
				};

				self.states[offset] = event.state == ElementState::Pressed;
			},
			WindowEvent::MouseWheel { delta, .. } => {
				let delta = match delta {
					MouseScrollDelta::LineDelta(_, y) => *y,
					MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
				};
				let factor = 2f32.powf(delta);
				self.move_speed *= factor;
			},
			_ => {},
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
			near: 0.001,
		}
	}
}
