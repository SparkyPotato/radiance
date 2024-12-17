use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable};
use rad_core::asset::aref::ARef;
use static_assertions::const_assert_eq;
use vek::{Aabb, Sphere, Vec2, Vec3};

use crate::assets::material::Material;

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

#[derive(Copy, Clone, Encode, Decode, Default)]
pub struct BvhNode {
	#[bincode(with_serde)]
	pub aabbs: [Aabb<f32>; 8],
	#[bincode(with_serde)]
	pub lod_bounds: [Sphere<f32, f32>; 8],
	pub parent_errors: [f32; 8],
	pub child_offsets: [u32; 8],
	pub child_counts: [u8; 8],
}

#[derive(Copy, Clone, Default, Encode, Decode)]
pub struct Meshlet {
	/// Offset of the meshlet vertex buffer relative to the parent mesh vertex buffer.
	pub vert_offset: u32,
	/// Offset of the meshlet index buffer relative to the parent mesh index buffer.
	pub index_offset: u32,
	/// Number of vertices in the meshlet.
	pub vert_count: u8,
	/// Number of triangles in the meshlet. The number of indices will be 3 times this.
	pub tri_count: u8,
	/// The AABB of the meshlet.
	#[bincode(with_serde)]
	pub aabb: Aabb<f32>,
	/// The bounds to use for LOD decisions.
	#[bincode(with_serde)]
	pub lod_bounds: Sphere<f32, f32>,
	/// The error of this meshlet.
	pub error: f32,
	/// The length of the longest edge in this meshlet.
	pub max_edge_length: f32,
}

/// A mesh.
#[derive(Encode, Decode)]
pub struct Mesh {
	/// Vertices of the mesh.
	pub vertices: Vec<Vertex>,
	/// Indices of each meshlet - should be added to `vertex_offset`.
	pub indices: Vec<u8>,
	/// Meshlets of the mesh.
	pub meshlets: Vec<Meshlet>,
	/// The LOD BVH of the mesh.
	pub bvh: Vec<BvhNode>,
	/// The max depth of the BVH.
	pub bvh_depth: u32,
	/// The AABB of the entire mesh.
	#[bincode(with_serde)]
	pub aabb: Aabb<f32>,
	/// Input vertices for the mesh.
	pub raw_vertices: Vec<Vertex>,
	/// Input indices for the mesh.
	pub raw_indices: Vec<u32>,
	#[bincode(with_serde)]
	/// Material of the mesh.
	pub material: ARef<Material>,
}
