use bincode::{Decode, Encode};
use uuid::Uuid;
use vek::Mat4;

#[derive(Encode, Decode)]
pub struct Node {
	pub name: String,
	#[bincode(with_serde)]
	pub transform: Mat4<f32>,
	#[bincode(with_serde)]
	pub model: Uuid,
}

#[derive(Encode, Decode)]
pub enum Projection {
	Perspective { yfov: f32, near: f32, far: Option<f32> },
	Orthographic { height: f32, near: f32, far: f32 },
}

#[derive(Encode, Decode)]
pub struct Camera {
	pub name: String,
	#[bincode(with_serde)]
	pub view: Mat4<f32>,
	pub projection: Projection,
}

#[derive(Encode, Decode)]
pub struct Scene {
	pub nodes: Vec<Node>,
	pub cameras: Vec<Camera>,
}
