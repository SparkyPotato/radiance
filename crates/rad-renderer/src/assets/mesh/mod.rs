use std::{
	array,
	io::{self, BufReader},
	usize,
};

use bincode::error::{DecodeError, EncodeError};
use bytemuck::{NoUninit, Pod, Zeroable};
use rad_core::{
	asset::{Asset, AssetView, Uuid},
	uuid,
	Engine,
};
use rad_graph::{
	device::Device,
	resource::{Buffer, BufferDesc, GpuPtr, Resource as _},
};
use static_assertions::const_assert_eq;
use tracing::{span, Level};
use vek::{Aabb, Sphere, Vec3, Vec4};

pub use crate::assets::mesh::import::MeshData;
use crate::util::SliceWriter;

mod data;
mod import;

pub type GpuVertex = data::Vertex;

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuAabb {
	pub center: Vec3<f32>,
	pub half_extent: Vec3<f32>,
}
const_assert_eq!(std::mem::size_of::<GpuAabb>(), 24);
const_assert_eq!(std::mem::align_of::<GpuAabb>(), 4);

pub(super) fn map_aabb(aabb: Aabb<f32>) -> GpuAabb {
	GpuAabb {
		center: aabb.center(),
		half_extent: aabb.half_size().into(),
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuBvhNode {
	pub aabbs: [GpuAabb; 8],
	pub lod_bounds: [Vec4<f32>; 8],
	pub parent_errors: [f32; 8],
	pub child_offsets: [u32; 8],
	pub child_counts: [u8; 8],
}
const_assert_eq!(std::mem::size_of::<GpuBvhNode>(), 392);
const_assert_eq!(std::mem::align_of::<GpuBvhNode>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuMeshlet {
	pub aabb: GpuAabb,
	pub lod_bounds: Vec4<f32>,
	pub error: f32,
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	pub _pad: u16,
	pub max_edge_length: f32,
}
const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 60);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

pub(super) fn map_sphere(sphere: Sphere<f32, f32>) -> Vec4<f32> { sphere.center.with_w(sphere.radius) }

pub struct Mesh {
	buffer: Buffer,
	bvh_depth: u32,
	aabb: Aabb<f32>,
}

impl Mesh {
	pub fn import(name: &str, import: import::MeshData, mut into: Box<dyn AssetView>) -> Result<(), io::Error> {
		let data = import::import(name, import);
		into.clear()?;
		bincode::encode_into_std_write(data, &mut into.new_section()?, bincode::config::standard()).map_err(
			|x| match x {
				EncodeError::Io { inner, .. } => inner,
				_ => io::Error::new(io::ErrorKind::Other, "bincode error"),
			},
		)?;

		Ok(())
	}

	pub fn bvh_depth(&self) -> u32 { self.bvh_depth }

	pub fn aabb(&self) -> Aabb<f32> { self.aabb }

	pub fn gpu_ptr(&self) -> GpuPtr<u8> { self.buffer.ptr() }
}

impl Asset for Mesh {
	fn uuid() -> Uuid
	where
		Self: Sized,
	{
		uuid!("0ab1a518-ced8-41c9-ae55-9c208a461636")
	}

	fn load(mut data: Box<dyn AssetView>) -> Result<Self, io::Error>
	where
		Self: Sized,
	{
		let s = span!(Level::TRACE, "decode mesh");
		let _e = s.enter();

		let device: &Device = Engine::get().global();
		data.seek_begin()?;
		let m: data::Mesh =
			bincode::decode_from_reader(BufReader::new(data.read_section()?), bincode::config::standard()).map_err(
				|x| match x {
					DecodeError::Io { inner, .. } => inner,
					_ => io::Error::new(io::ErrorKind::Other, "bincode error"),
				},
			)?;

		let bvh_byte_offset = 0;
		let bvh_byte_len = (m.bvh.len() * std::mem::size_of::<GpuBvhNode>()) as u64;
		let meshlet_byte_offset = bvh_byte_offset + bvh_byte_len;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() * std::mem::size_of::<u8>()) as u64;
		let size = index_byte_offset + index_byte_len;

		let buffer = Buffer::create(
			device,
			BufferDesc {
				name: "mesh buffer",
				size,
				readback: false,
			},
		)
		.map_err(|x| io::Error::new(io::ErrorKind::Other, format!("failed to create mesh buffer: {:?}", x)))?;
		let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });

		for node in m.bvh {
			writer.write(GpuBvhNode {
				aabbs: node.aabbs.map(map_aabb),
				lod_bounds: node.lod_bounds.map(map_sphere),
				parent_errors: node.parent_errors,
				child_offsets: array::from_fn(|i| {
					if node.child_counts[i] == u8::MAX {
						bvh_byte_offset as u32 + node.child_offsets[i] * std::mem::size_of::<GpuBvhNode>() as u32
					} else {
						meshlet_byte_offset as u32 + node.child_offsets[i] * std::mem::size_of::<GpuMeshlet>() as u32
					}
				}),
				child_counts: node.child_counts,
			});
		}

		for me in m.meshlets.iter() {
			writer.write(GpuMeshlet {
				aabb: map_aabb(me.aabb),
				lod_bounds: map_sphere(me.lod_bounds),
				error: me.error,
				vertex_byte_offset: vertex_byte_offset as u32
					+ (me.vert_offset * std::mem::size_of::<GpuVertex>() as u32),
				index_byte_offset: index_byte_offset as u32 + (me.index_offset * std::mem::size_of::<u8>() as u32),
				vertex_count: me.vert_count,
				triangle_count: me.tri_count,
				_pad: 0,
				max_edge_length: me.max_edge_length,
			});
		}

		writer.write_slice(&m.vertices);
		writer.write_slice(&m.indices);

		Ok(Mesh {
			buffer,
			bvh_depth: m.bvh_depth,
			aabb: m.aabb,
		})
	}

	fn save(&self, _: &mut dyn AssetView) -> Result<(), io::Error> {
		Err(io::Error::new(io::ErrorKind::Unsupported, "meshes cannot be edited"))
	}
}
