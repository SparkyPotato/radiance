use rad_world::{bevy_reflect::Reflect, RadComponent};
use vek::Vec3;

#[derive(Copy, Clone, Reflect)]
pub enum LightType {
	Point,
	Directional,
}

#[derive(RadComponent)]
#[uuid("69a570e9-032e-4ca0-aa96-92e9cc4a950c")]
pub struct LightComponent {
	pub color: Vec3<f32>,
	pub intensity: f32,
	pub ty: LightType,
}
