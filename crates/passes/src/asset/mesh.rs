use std::usize;

use ash::vk;
use bytemuck::{NoUninit, Pod, Zeroable};
use crossbeam_channel::Sender;
use radiance_asset::{mesh::Vertex, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	graph::Resource,
	resource::{Buffer, BufferDesc, Resource as _},
};
use static_assertions::const_assert_eq;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Aabb, Ray, Sphere, Vec3, Vec4};

use crate::asset::{
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	LoadError,
	Loader,
};

pub type GpuVertex = Vertex;

#[derive(Copy, Clone, Pod, Zeroable)]
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
	pub aabb: GpuAabb,
	pub lod_bounds: Vec4<f32>,
	pub parent_error: f32,
	pub children_offset: u32,
	pub child_count: u32,
}
const_assert_eq!(std::mem::size_of::<GpuBvhNode>(), 52);
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
}
const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 56);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

pub(super) fn map_sphere(sphere: Sphere<f32, f32>) -> Vec4<f32> { sphere.center.with_w(sphere.radius) }

pub struct Mesh {
	pub(super) buffer: Buffer,
	pub(super) bvh_depth: u32,
	pub(super) aabb: Aabb<f32>,
	// pub(super) vertices: Vec<Vertex>,
	// pub(super) mesh: TriMesh,
}

impl Mesh {
	pub fn intersect(&self, _: Ray<f32>, _: f32) -> bool {
		// let s = span!(Level::TRACE, "mesh intersect");
		// let _e = s.enter();
		//
		// let pray = parry3d::query::Ray {
		// 	origin: Point::new(ray.origin.x, ray.origin.y, ray.origin.z),
		// 	dir: Vector::new(ray.direction.x, ray.direction.y, ray.direction.z),
		// };
		// self.mesh.cast_local_ray_and_get_normal(&pray, tmax, true).map(|int| {
		// 	let t = int.time_of_impact;
		// 	let mut trii = int.feature.unwrap_face();
		// 	if trii >= self.mesh.num_triangles() as u32 {
		// 		trii -= self.mesh.num_triangles() as u32;
		// 	}
		//
		// 	let tri = self.mesh.triangle(trii);
		// 	let position = pray.origin + pray.dir * t;
		// 	let v0 = tri.b - tri.a;
		// 	let v1 = tri.c - tri.a;
		// 	let v2 = position - tri.a;
		// 	let d00 = v0.dot(&v0);
		// 	let d01 = v0.dot(&v1);
		// 	let d11 = v1.dot(&v1);
		// 	let d20 = v2.dot(&v0);
		// 	let d21 = v2.dot(&v1);
		// 	let denom = d00 * d11 - d01 * d01;
		// 	let v = (d11 * d20 - d01 * d21) / denom;
		// 	let w = (d00 * d21 - d01 * d20) / denom;
		// 	let u = 1.0 - v - w;
		//
		// 	let i = self.mesh.indices()[trii as usize];
		// 	let v0 = self.vertices[i[0] as usize];
		// 	let v1 = self.vertices[i[1] as usize];
		// 	let v2 = self.vertices[i[2] as usize];
		//
		// 	let normal = v0.normal * u + v1.normal * v + v2.normal * w;
		// 	let uv = v0.uv * u + v1.uv * v + v2.uv * w;
		//
		// 	Intersection {
		// 		t,
		// 		position: Vec3::new(position.x, position.y, position.z),
		// 		normal,
		// 		uv,
		// 	}
		// })
		false
	}
}

impl RuntimeAsset for Mesh {
	fn into_resources(self, queue: Sender<DelRes>) { queue.send(Resource::Buffer(self.buffer).into()).unwrap(); }
}

impl<S: AssetSource> Loader<'_, S> {
	pub fn load_mesh(&mut self, uuid: Uuid) -> LResult<Mesh, S> {
		match AssetRuntime::get_cache(&mut self.runtime.meshes, uuid) {
			Some(x) => Ok(x),
			None => {
				let m = self.load_mesh_from_disk(uuid)?;
				self.runtime.meshes.insert(uuid, m.downgrade());
				Ok(m)
			},
		}
	}

	fn load_mesh_from_disk(&self, mesh: Uuid) -> LResult<Mesh, S> {
		let s = span!(Level::TRACE, "load mesh from disk");
		let _e = s.enter();

		let Asset::Mesh(m) = self.sys.load(mesh)? else {
			unreachable!("Mesh asset is not a mesh");
		};

		let bvh_byte_offset = 0;
		let bvh_byte_len = (m.bvh.len() * std::mem::size_of::<GpuBvhNode>()) as u64;
		let meshlet_byte_offset = bvh_byte_offset + bvh_byte_len;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() * std::mem::size_of::<u8>()) as u64;
		let size = index_byte_offset + index_byte_len;

		let name = self.sys.human_name(mesh).unwrap_or("unnamed mesh".to_string());
		let buffer = Buffer::create(
			self.device,
			BufferDesc {
				name: &format!("{name} buffer"),
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
				on_cpu: false,
			},
		)
		.map_err(LoadError::Vulkan)?;

		// let vertices: Vec<_> = m
		// 	.vertices
		// 	.iter()
		// 	.map(|v| Point::new(v.position.x, v.position.y, v.position.z))
		// 	.collect();

		let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });

		for node in m.bvh {
			let is_meshlet = (node.child_count >> 7) == 1;
			writer
				.write(GpuBvhNode {
					aabb: map_aabb(node.aabb),
					lod_bounds: map_sphere(node.lod_bounds),
					parent_error: node.parent_error,
					children_offset: if is_meshlet {
						meshlet_byte_offset as u32 + node.children_offset * std::mem::size_of::<GpuMeshlet>() as u32
					} else {
						bvh_byte_offset as u32 + node.children_offset * std::mem::size_of::<GpuBvhNode>() as u32
					},
					child_count: node.child_count as _,
				})
				.unwrap();
		}

		// let indices = m
		// 	.meshlets
		// 	.iter()
		// 	.flat_map(|me| {
		// 		let tc = me.tri_count as usize;
		// 		(0..tc).map(|t| {
		// 			let i0 = me.index_offset as usize + t * 3;
		// 			let i1 = i0 + 1;
		// 			let i2 = i1 + 1;
		// 			[
		// 				m.indices[i0] as u32 + me.vertex_offset,
		// 				m.indices[i1] as u32 + me.vertex_offset,
		// 				m.indices[i2] as u32 + me.vertex_offset,
		// 			]
		// 		})
		// 	})
		// 	.collect();

		// let mesh = TriMesh::new(vertices, indices);

		for me in m.meshlets.iter() {
			writer
				.write(GpuMeshlet {
					aabb: map_aabb(me.aabb),
					lod_bounds: map_sphere(me.lod_bounds),
					error: me.error,
					vertex_byte_offset: vertex_byte_offset as u32
						+ (me.vert_offset * std::mem::size_of::<GpuVertex>() as u32),
					index_byte_offset: index_byte_offset as u32 + (me.index_offset * std::mem::size_of::<u8>() as u32),
					vertex_count: me.vert_count,
					triangle_count: me.tri_count,
					_pad: 0,
				})
				.unwrap();
		}

		writer.write_slice(&m.vertices).unwrap();
		writer.write_slice(&m.indices).unwrap();

		Ok(RRef::new(
			Mesh {
				buffer,
				bvh_depth: m.bvh_depth,
				aabb: m.aabb,
				// vertices: m.vertices,
				// mesh,
			},
			self.runtime.deleter.clone(),
		))
	}
}
