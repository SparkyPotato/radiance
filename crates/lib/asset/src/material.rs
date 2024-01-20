use bincode::{Decode, Encode};
use uuid::Uuid;
use vek::{Vec3, Vec4};

#[derive(Copy, Clone, Eq, PartialEq, Encode, Decode)]
pub enum AlphaMode {
	Opaque,
	Mask,
	Blend,
}

#[derive(Encode, Decode)]
pub struct Material {
	pub alpha_cutoff: f32,
	pub alpha_mode: AlphaMode,
	#[bincode(with_serde)]
	pub base_color_factor: Vec4<f32>,
	#[bincode(with_serde)]
	pub base_color: Option<Uuid>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	#[bincode(with_serde)]
	pub metallic_roughness: Option<Uuid>,
	#[bincode(with_serde)]
	pub normal: Option<Uuid>,
	#[bincode(with_serde)]
	pub occlusion: Option<Uuid>,
	#[bincode(with_serde)]
	pub emissive_factor: Vec3<f32>,
	#[bincode(with_serde)]
	pub emissive: Option<Uuid>,
}
