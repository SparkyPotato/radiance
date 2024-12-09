use vek::{Quaternion, Vec3};

use crate::{rad_world, RadComponent};

#[derive(RadComponent)]
#[uuid("efcddf51-d15c-434b-bff4-1a0fe18ba53b")]
pub struct Transform {
	pub position: Vec3<f32>,
	pub rotation: Quaternion<f32>,
	pub scale: Vec3<f32>,
}

impl Transform {
	pub fn identity() -> Self {
		Self {
			position: Vec3::zero(),
			rotation: Quaternion::identity(),
			scale: Vec3::broadcast(1.0),
		}
	}
}
