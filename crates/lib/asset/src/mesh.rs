use std::ops::Range;

use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Sphere, Vec2, Vec3};

#[derive(Pod, Zeroable, Copy, Clone, Default, Encode, Decode)]
#[repr(C)]
pub struct Vertex {
	#[bincode(with_serde)]
	pub position: Vec3<f32>,
	#[bincode(with_serde)]
	pub normal: Vec3<f32>,
	#[bincode(with_serde)]
	pub uv: Vec2<f32>,
}

const_assert_eq!(std::mem::size_of::<Vertex>(), 32);
const_assert_eq!(std::mem::align_of::<Vertex>(), 4);

#[derive(Encode, Decode)]
pub struct Meshlet {
	/// Offset of the meshlet vertex buffer relative to the parent mesh vertex buffer.
	pub vertex_offset: u32,
	/// Offset of the meshlet index buffer relative to the parent mesh index buffer.
	pub index_offset: u32,
	/// Number of vertices in the meshlet.
	pub vert_count: u8,
	/// Number of triangles in the meshlet. The number of indices will be 3 times this.
	pub tri_count: u8,
	/// The materials assigned to triangles in this meshlet.
	pub material_ranges: Range<u32>,
	/// The bounding sphere of the meshlet.
	#[bincode(with_serde)]
	pub bounding: Sphere<f32, f32>,
	/// The error sphere of the meshlet group, used for LOD decision.
	#[bincode(with_serde)]
	pub group_error: Sphere<f32, f32>,
	/// The error sphere of the parent meshlet group, used for LOD decision.
	#[bincode(with_serde)]
	pub parent_group_error: Sphere<f32, f32>,
}

#[derive(Encode, Decode)]
pub struct MaterialRange {
	/// The material referenced.
	#[bincode(with_serde)]
	pub material: Uuid,
	/// The vertices that the material is applied to.
	pub vertices: Range<u8>,
}

/// A mesh consisting of multiple submeshes, each with a material assigned to it.
#[derive(Encode, Decode)]
pub struct Mesh {
	/// Vertices of the mesh.
	pub vertices: Vec<Vertex>,
	/// Indices of each meshlet - should be added to `vertex_offset`.
	pub indices: Vec<u8>,
	/// Meshlets of the mesh.
	pub meshlets: Vec<Meshlet>,
	/// Ranges of materials in each meshlet.
	pub material_ranges: Vec<MaterialRange>,
}
