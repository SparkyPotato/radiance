use bytemuck::{Pod, Zeroable};
use static_assertions::const_assert_eq;
use vek::{Aabb, Vec2, Vec3, Vec4};

use crate::util::{SliceReader, SliceWriter};

#[derive(Pod, Zeroable, Copy, Clone, Default)]
#[repr(C)]
pub struct Vertex {
	/// Normalized vertex coordinates relative to the meshlet AABB.
	pub position: Vec3<u16>,
	/// Signed normalized normal vector.
	pub normal: Vec3<i16>,
	/// Normalized UV coordinates relative to the [0.0, 1.0] UV range.
	pub uv: Vec2<u16>,
}

const_assert_eq!(std::mem::size_of::<Vertex>(), 16);
const_assert_eq!(std::mem::align_of::<Vertex>(), 2);

/// Culling cone.
#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct Cone {
	pub apex: Vec4<u8>,
	/// Signed normalized axis of the cone.
	pub axis: Vec3<i8>,
	/// Signed normalized cos(angle / 2).
	pub cutoff: i8,
}

const_assert_eq!(std::mem::size_of::<Cone>(), 8);
const_assert_eq!(std::mem::align_of::<Cone>(), 1);

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct Meshlet {
	/// AABB of the meshlet relative to the mesh origin.
	pub aabb_min: Vec3<f32>,
	pub aabb_extent: Vec3<f32>,
	/// Offset of the meshlet index buffer relative to the parent mesh index buffer.
	pub index_offset: u32,
	/// Offset of the meshlet vertex buffer relative to the parent mesh vertex buffer.
	pub vertex_offset: u32,
	/// Number of triangles in the meshlet. The number of indices will be 3 times this.
	pub tri_count: u8,
	/// Number of vertices in the meshlet.
	pub vert_count: u8,
	/// Cone of the meshlet relative to the center of the bounding box.
	pub cone: Cone,
	pub _pad: u16,
}

const_assert_eq!(std::mem::size_of::<Meshlet>(), 44);
const_assert_eq!(std::mem::align_of::<Meshlet>(), 4);

/// A mesh asset consisting of meshlets.
pub struct Mesh {
	/// Vertices of the mesh.
	pub vertices: Vec<Vertex>,
	/// Indices of each meshlet - should be added to `vertex_offset`.
	pub indices: Vec<u8>,
	/// Meshlets of the mesh.
	pub meshlets: Vec<Meshlet>,
	/// AABB of the mesh.
	pub aabb: Aabb<f32>,
}

impl Mesh {
	/// - AABB.
	/// - 5 u32s: vertex count, index count, meshlet count, vertex bytes, index bytes.
	/// - meshopt encoded vertex buffer.
	/// - meshopt encoded index buffer.
	/// - padding for alignment.
	/// - meshlets.
	///
	/// Everything is little endian, compressed by zstd.
	pub(super) fn to_bytes(&self) -> Vec<u8> {
		let vertices = meshopt::encode_vertex_buffer(&self.vertices).unwrap();
		let indices: Vec<_> = self.indices.iter().map(|&x| x as u32).collect();
		let indices = meshopt::encode_index_buffer(&indices, self.vertices.len()).unwrap();

		let vertex_len = vertices.len();
		let index_len = indices.len();
		let meshlet_len = std::mem::size_of::<Meshlet>() * self.meshlets.len();

		let extra = (vertex_len + index_len) % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		let len = std::mem::size_of::<Vec3<f32>>() * 2 + 4 * 5 + vertex_len + index_len + meshlet_len + fill;
		let mut bytes = vec![0; len];
		let mut writer = SliceWriter::new(bytes.as_mut_slice());

		writer.write(self.aabb.min);
		writer.write(self.aabb.max);
		writer.write(self.vertices.len() as u32);
		writer.write(self.indices.len() as u32);
		writer.write(self.meshlets.len() as u32);
		writer.write(vertex_len as u32);
		writer.write(index_len as u32);

		writer.write_slice(&vertices);
		writer.write_slice(&indices);
		writer.write_slice(&[0u8, 0, 0][0..fill]);
		writer.write_slice(&self.meshlets);

		zstd::encode_all(bytes.as_slice(), 8).unwrap()
	}

	pub(super) fn from_bytes(bytes: &[u8]) -> Self {
		let bytes = zstd::decode_all(bytes).unwrap();
		let mut reader = SliceReader::new(&bytes);

		let min = reader.read::<Vec3<f32>>();
		let max = reader.read::<Vec3<f32>>();
		let vertex_count = reader.read::<u32>() as usize;
		let index_count = reader.read::<u32>() as usize;
		let meshlet_count = reader.read::<u32>() as usize;
		let vertex_len = reader.read::<u32>() as usize;
		let index_len = reader.read::<u32>() as usize;

		let extra = (vertex_len + index_len) % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };

		let vertices = meshopt::decode_vertex_buffer(reader.read_slice(vertex_len), vertex_count).unwrap();
		let indices: Vec<u32> = meshopt::decode_index_buffer(reader.read_slice(index_len), index_count).unwrap();
		let indices: Vec<_> = indices.into_iter().map(|x| x as u8).collect();
		reader.read_slice::<u8>(fill);
		let meshlets = reader.read_slice(meshlet_count).to_vec();

		Self {
			vertices,
			indices,
			meshlets,
			aabb: Aabb { min, max },
		}
	}
}
