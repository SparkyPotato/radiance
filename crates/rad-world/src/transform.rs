use vek::{Mat4, Quaternion, Vec3};

use crate::{rad_world, RadComponent};

#[derive(Copy, Clone, Debug, PartialEq, RadComponent)]
#[uuid("efcddf51-d15c-434b-bff4-1a0fe18ba53b")]
pub struct Transform {
	pub position: Vec3<f32>,
	pub rotation: Quaternion<f32>,
	pub scale: Vec3<f32>,
}

impl Default for Transform {
	fn default() -> Self { Self::identity() }
}

impl Transform {
	pub fn identity() -> Self {
		Self {
			position: Vec3::zero(),
			rotation: Quaternion::identity(),
			scale: Vec3::broadcast(1.0),
		}
	}

	pub fn into_matrix(self) -> Mat4<f32> {
		let (angle, axis) = self.rotation.into_angle_axis();
		Mat4::identity()
			.scaled_3d(self.scale)
			.rotated_3d(angle, axis)
			.translated_3d(self.position)
	}
}
