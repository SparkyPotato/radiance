use rad_world::RadComponent;

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
			fov: 70f32.to_radians(),
			near: 0.01,
		}
	}
}

pub use view::PrimaryViewComponent;

mod view {
	use super::*;

	#[derive(Copy, Clone, PartialEq, RadComponent)]
	#[uuid("201a2ef5-1bcc-442b-ad1d-1af7e7ac63e5")]
	pub struct PrimaryViewComponent;
}
