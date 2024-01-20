use bincode::{Decode, Encode};
use uuid::Uuid;
use vek::Aabb;

/// A model consisting of multiple meshes and a material assigned to each mesh.
#[derive(Clone, Encode, Decode)]
pub struct Model {
	#[bincode(with_serde)]
	pub meshes: Vec<Uuid>,
	#[bincode(with_serde)]
	pub aabb: Aabb<f32>,
}
