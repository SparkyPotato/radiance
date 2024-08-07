use std::usize;

use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use parry3d::{
	math::{Point, Vector},
	query::RayCast,
	shape::TriMesh,
};
use radiance_asset::{mesh::Vertex, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	device::Compute,
	graph::Resource,
	resource::{ASDesc, Buffer, BufferDesc, Resource as _, AS},
};
use static_assertions::const_assert_eq;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Ray, Vec2, Vec3};

use crate::asset::{
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	LoadError,
	Loader,
};

pub type GpuVertex = Vertex;

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Sphere {
	center: Vec3<f32>,
	radius: f32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuMeshlet {
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	pub _pad: u16,
	pub bounding: Sphere,
	pub group_error: Sphere,
	pub parent_error: Sphere,
}

const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 60);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

pub struct Mesh {
	pub(super) buffer: Buffer,
	pub(super) acceleration_structure: AS,
	pub(super) meshlet_count: u32,
	// pub(super) vertices: Vec<Vertex>,
	// pub(super) mesh: TriMesh,
}

#[derive(Copy, Clone)]
pub struct Intersection {
	pub t: f32,
	pub position: Vec3<f32>,
	pub normal: Vec3<f32>,
	pub uv: Vec2<f32>,
}

impl Mesh {
	pub fn intersect(&self, _: Ray<f32>, _: f32) -> Option<Intersection> {
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
		None
	}
}

impl RuntimeAsset for Mesh {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(Resource::Buffer(self.buffer).into()).unwrap();
		queue.send(Resource::AS(self.acceleration_structure).into()).unwrap();
	}
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

		let meshlet_byte_offset = 0;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() / 3 * std::mem::size_of::<u32>()) as u64;
		let size = index_byte_offset + index_byte_len;
		let meshlet_count = m.meshlets.len() as u32;

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

		// let raw_mesh = Buffer::create(
		// 	self.device,
		// 	BufferDesc {
		// 		name: "AS build scratch mesh",
		// 		size: (std::mem::size_of::<u16>() * m.indices.len()) as u64,
		// 		usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
		// 			| vk::BufferUsageFlags::STORAGE_BUFFER,
		// 		on_cpu: false,
		// 	},
		// )
		// .map_err(LoadError::Vulkan)?;
		// let mut iwriter = SliceWriter::new(unsafe { raw_mesh.data().as_mut() });
		// let mut geo = Vec::new();
		// let mut counts = Vec::new();
		// let mut ranges = Vec::new();

		for me in m.meshlets.iter() {
			// let off = me.index_offset as usize;
			// for &i in m.indices[off..off + me.tri_count as usize * 3].iter() {
			// 	iwriter.write(i as u16).unwrap();
			// }

			// let stride = std::mem::size_of::<GpuVertex>() as u64;
			// geo.push(
			// 	vk::AccelerationStructureGeometryKHR::default()
			// 		.geometry_type(vk::GeometryTypeKHR::TRIANGLES)
			// 		.geometry(vk::AccelerationStructureGeometryDataKHR {
			// 			triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
			// 				.vertex_format(vk::Format::R32G32B32_SFLOAT)
			// 				.vertex_data(vk::DeviceOrHostAddressConstKHR {
			// 					device_address: buffer.addr() + vertex_byte_offset,
			// 				})
			// 				.vertex_stride(stride)
			// 				.max_vertex(me.tri_count as u32 - 1)
			// 				.index_type(vk::IndexType::UINT16)
			// 				.index_data(vk::DeviceOrHostAddressConstKHR {
			// 					device_address: raw_mesh.addr()
			// 						+ me.index_offset as u64 * std::mem::size_of::<u16>() as u64,
			// 				}),
			// 		})
			// 		.flags(vk::GeometryFlagsKHR::OPAQUE),
			// );
			// counts.push(me.tri_count as u32);
			// ranges.push(vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(me.tri_count as u32));

			writer
				.write(GpuMeshlet {
					vertex_byte_offset: vertex_byte_offset as u32
						+ (me.vertex_offset * std::mem::size_of::<GpuVertex>() as u32),
					index_byte_offset: index_byte_offset as u32
						+ (me.index_offset / 3 * std::mem::size_of::<u32>() as u32),
					vertex_count: me.vert_count,
					triangle_count: me.tri_count,
					_pad: 0,
					bounding: Sphere {
						center: me.bounding.center,
						radius: me.bounding.radius,
					},
					group_error: Sphere {
						center: me.group_bounding.center,
						radius: me.group_bounding.radius,
					},
					parent_error: Sphere {
						center: me.parent_group_bounding.center,
						radius: me.parent_group_bounding.radius,
					},
				})
				.unwrap();
		}

		writer.write_slice(&m.vertices).unwrap();

		for tri in m.indices.chunks(3) {
			writer.write_slice(tri).unwrap();
			writer.write(0u8).unwrap();
		}

		let acceleration_structure = unsafe {
			// let ext = self.device.as_ext();
			//
			// let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
			// 	.ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
			// 	.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
			// 	.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
			// 	// .geometries(&geo);
			// 	.geometries(&[]);
			//
			// let mut size = Default::default();
			// ext.get_acceleration_structure_build_sizes(
			// 	vk::AccelerationStructureBuildTypeKHR::DEVICE,
			// 	&info,
			// 	// &counts,
			// 	&[],
			// 	&mut size,
			// );

			// let as_ = AS::create(
			// 	self.device,
			// 	ASDesc {
			// 		name: &format!("{name} AS"),
			// 		flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
			// 		ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
			// 		size: size.acceleration_structure_size,
			// 	},
			// )
			// .map_err(LoadError::Vulkan)?;

			// let scratch = Buffer::create(
			// 	self.device,
			// 	BufferDesc {
			// 		name: "AS build scratch",
			// 		size: size.build_scratch_size,
			// 		usage: vk::BufferUsageFlags::STORAGE_BUFFER
			// 			| vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
			// 		on_cpu: false,
			// 	},
			// )
			// .map_err(LoadError::Vulkan)?;
			//
			// info.dst_acceleration_structure = as_.handle();
			// info.scratch_data = vk::DeviceOrHostAddressKHR {
			// 	device_address: scratch.addr(),
			// };

			// let buf = self.ctx.get_buf::<Compute>();
			// ext.cmd_build_acceleration_structures(buf, &[info], &[&ranges]);
			//
			// self.ctx.delete::<Compute>(scratch);
			// self.ctx.delete::<Compute>(raw_mesh);

			// as_
			AS::default()
		};

		Ok(RRef::new(
			Mesh {
				buffer,
				meshlet_count,
				acceleration_structure,
				// vertices: m.vertices,
				// mesh,
			},
			self.runtime.deleter.clone(),
		))
	}
}
