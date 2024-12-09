use rad_world::RadComponent;
use vek::Mat4;

#[derive(Copy, Clone, PartialEq, RadComponent)]
#[uuid("34262fdf-3f97-47ab-a42a-a89786d6b2ac")]
pub struct CameraComponent {
	/// Vertical FOV in radians.
	pub fov: f32,
	pub near: f32,
}

impl Default for CameraComponent {
	fn default() -> Self {
		Self {
			fov: std::f32::consts::PI / 2.0,
			near: 0.01,
		}
	}
}

impl CameraComponent {
	pub fn projection(&self, aspect: f32) -> Mat4<f32> {
		let h = (self.fov / 2.0).tan().recip();
		let w = h / aspect;
		let near = self.near;
		Mat4::new(
			w, 0.0, 0.0, 0.0, //
			0.0, h, 0.0, 0.0, //
			0.0, 0.0, 0.0, near, //
			0.0, 0.0, 1.0, 0.0, //
		)
	}
}

pub use view::PrimaryViewComponent;

mod view {
	use super::*;

	#[derive(Copy, Clone, PartialEq, RadComponent)]
	#[uuid("5bbc8e09-36c5-4283-863e-9685a2c91c70")]
	pub struct PrimaryViewComponent(pub CameraComponent);
}
