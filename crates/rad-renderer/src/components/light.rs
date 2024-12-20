use rad_world::{bevy_reflect::Reflect, RadComponent};
use vek::Vec3;

#[derive(Copy, Clone, Reflect)]
pub enum LightType {
	Point,
	Directional,
	Sky,
}

#[derive(RadComponent)]
#[uuid("69a570e9-032e-4ca0-aa96-92e9cc4a950c")]
pub struct LightComponent {
	pub ty: LightType,
	pub radiance: Vec3<f32>,
}
