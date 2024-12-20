use rad_renderer::vek::{num_traits::FloatConst, Mat4, Quaternion, Vec2, Vec3, Vec4};
use rad_ui::egui::Context;
use rad_window::winit::{
	dpi::PhysicalPosition,
	event::{ElementState, MouseScrollDelta, WindowEvent},
	keyboard::{KeyCode, PhysicalKey},
	window::{CursorGrabMode, Window},
};
use rad_world::{transform::Transform, EntityWrite};

#[derive(Default)]
struct MouseGrabber {
	last_pos: PhysicalPosition<f64>,
	manual_lock: bool,
}

impl MouseGrabber {
	fn cursor_moved(&mut self, window: &Window, pos: PhysicalPosition<f64>) {
		if self.manual_lock {
			window.set_cursor_position(self.last_pos).unwrap();
		} else {
			self.last_pos = pos;
		}
	}

	fn grab(&mut self, window: &Window, grab: bool) {
		if grab {
			// TODO: winit bug stops our events if confined
			// window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
			self.manual_lock = true;
		} else {
			self.manual_lock = false;
			window.set_cursor_grab(CursorGrabMode::None).unwrap();
		}
		window.set_cursor_visible(!grab);
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

		let yaw = Mat4::identity().rotated_z(self.yaw);
		let forward = (yaw * Vec4::unit_y()).xyz();
		let right = (yaw * Vec4::unit_x()).xyz();
		let states = [forward, -forward, right, -right, Vec3::unit_z(), -Vec3::unit_z()];
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
		self.pos += dir * self.move_speed * ctx.input(|x| x.unstable_dt);
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		match event {
			WindowEvent::CursorMoved { position, .. } => {
				self.grabber.cursor_moved(window, *position);
				let delta = Vec2::new(position.x as f32, position.y as _)
					- Vec2::new(self.grabber.last_pos.x as _, self.grabber.last_pos.y as _);
				let delta = Vec2::new(delta.x, delta.y) * 0.002;
				self.pitch -= delta.y;
				self.yaw -= delta.x;
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
			WindowEvent::MouseWheel { delta, .. } if self.mode == Mode::Camera => {
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

	pub fn apply(&self, mut entity: EntityWrite<'_>) {
		let t = entity.component_mut::<Transform>().unwrap();
		t.position = self.pos;
		t.rotation = Quaternion::identity().rotated_x(self.pitch).rotated_z(self.yaw);
	}
}
