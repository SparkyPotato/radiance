use std::usize;

use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use radiance_asset::{mesh::Vertex, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	device::QueueType,
	resource::{ASDesc, BufferDesc, GpuBuffer, Resource, AS},
};
use radiance_util::{deletion::IntoResource, staging::StageError};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::Vec3;

use crate::{
	material::Material,
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LErr,
	LResult,
	Loader,
};

pub type GpuVertex = Vertex;

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuMeshlet {
	pub aabb_min: Vec3<f32>,
	pub aabb_extent: Vec3<f32>,
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	pub submesh: u16,
}

const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 36);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuSubMesh {
	pub material: u32,
}

const_assert_eq!(std::mem::size_of::<GpuSubMesh>(), 4);
const_assert_eq!(std::mem::align_of::<GpuSubMesh>(), 4);

pub struct Mesh {
	pub buffer: GpuBuffer,
	pub submeshes: Vec<RRef<Material>>,
	pub acceleration_structure: AS,
	pub meshlet_count: u32,
}

impl RuntimeAsset for Mesh {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(self.buffer.into_resource().into()).unwrap();
		queue.send(self.acceleration_structure.into_resource().into()).unwrap();
	}
}

impl AssetRuntime {
	pub(crate) fn load_mesh_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, '_, '_, S>, mesh: Uuid,
	) -> LResult<Mesh, S> {
		let Asset::Mesh(m) = loader.sys.load(mesh)? else {
			unreachable!("Mesh asset is not a mesh");
		};

		let submesh_byte_offset = 0;
		let submesh_byte_len = (m.submeshes.len() * std::mem::size_of::<GpuSubMesh>()) as u64;
		let meshlet_byte_offset = submesh_byte_offset + submesh_byte_len;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() / 3 * std::mem::size_of::<u32>()) as u64;
		let size = index_byte_offset + index_byte_len;
		let meshlet_count = m.meshlets.len() as u32;

		let buffer = GpuBuffer::create(
			loader.device,
			BufferDesc {
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			},
		)
		.map_err(StageError::Vulkan)?;

		let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });
		let submeshes = m
			.submeshes
			.iter()
			.map(|x| {
				let mat = self.load_material(loader, x.material)?;
				writer.write(GpuSubMesh { material: mat.index }).unwrap();
				Ok(mat)
			})
			.collect::<Result<_, LErr<S>>>()?;

		let vertex_size = (std::mem::size_of::<Vec3<f32>>() * m.vertices.len()) as u64;
		let raw_mesh = GpuBuffer::create(
			loader.device,
			BufferDesc {
				size: vertex_size + (std::mem::size_of::<u16>() * m.indices.len()) as u64,
				usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
					| vk::BufferUsageFlags::STORAGE_BUFFER,
			},
		)
		.map_err(StageError::Vulkan)?;
		let mut vwriter = SliceWriter::new(unsafe { &mut raw_mesh.data().as_mut()[..vertex_size as usize] });
		let mut iwriter = SliceWriter::new(unsafe { &mut raw_mesh.data().as_mut()[vertex_size as usize..] });
		let mut geo = Vec::with_capacity(m.submeshes.len());
		let mut counts = Vec::with_capacity(m.submeshes.len());
		let mut ranges = Vec::with_capacity(m.submeshes.len());

		for (s, sub) in m.submeshes.iter().enumerate() {
			for me in sub.meshlets.clone().map(|i| &m.meshlets[i as usize]) {
				let aabb_extent = me.aabb.max - me.aabb.min;
				let off = me.vertex_offset as usize;
				for v in m.vertices[off..off + me.vert_count as usize].iter() {
					vwriter
						.write(me.aabb.min + aabb_extent * v.position.map(|x| x as f32 / u16::MAX as f32))
						.unwrap();
				}
				let off = me.index_offset as usize;
				for &i in m.indices[off..off + me.tri_count as usize * 3].iter() {
					iwriter.write(i as u16).unwrap();
				}

				let stride = std::mem::size_of::<Vec3<f32>>() as u64;
				geo.push(
					vk::AccelerationStructureGeometryKHR::builder()
						.geometry_type(vk::GeometryTypeKHR::TRIANGLES)
						.geometry(vk::AccelerationStructureGeometryDataKHR {
							triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
								.vertex_format(vk::Format::R32G32B32_SFLOAT)
								.vertex_data(vk::DeviceOrHostAddressConstKHR {
									device_address: raw_mesh.addr() + me.vertex_offset as u64 * stride,
								})
								.vertex_stride(stride)
								.max_vertex(me.tri_count as u32 - 1)
								.index_type(vk::IndexType::UINT16)
								.index_data(vk::DeviceOrHostAddressConstKHR {
									device_address: raw_mesh.addr()
										+ vertex_size + me.index_offset as u64
										* std::mem::size_of::<u16>() as u64,
								})
								.build(),
						})
						.flags(vk::GeometryFlagsKHR::OPAQUE)
						.build(),
				);
				counts.push(me.tri_count as u32);
				ranges.push(
					vk::AccelerationStructureBuildRangeInfoKHR::builder()
						.primitive_count(me.tri_count as u32)
						.build(),
				);

				writer
					.write(GpuMeshlet {
						aabb_min: me.aabb.min,
						aabb_extent,
						vertex_byte_offset: vertex_byte_offset as u32
							+ (me.vertex_offset * std::mem::size_of::<GpuVertex>() as u32),
						index_byte_offset: index_byte_offset as u32
							+ (me.index_offset / 3 * std::mem::size_of::<u32>() as u32),
						vertex_count: me.vert_count,
						triangle_count: me.tri_count,
						submesh: s as u16,
					})
					.unwrap();
			}
		}

		writer.write_slice(&m.vertices).unwrap();

		for tri in m.indices.chunks(3) {
			writer.write_slice(tri).unwrap();
			writer.write(0u8).unwrap();
		}

		let acceleration_structure = unsafe {
			let ext = loader.device.as_ext();

			let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
				.ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
				.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
				.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
				.geometries(&geo);

			let size = ext.get_acceleration_structure_build_sizes(
				vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&info,
				&counts,
			);

			let as_ = AS::create(
				loader.device,
				ASDesc {
					flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
					ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
					size: size.acceleration_structure_size,
				},
			)
			.map_err(StageError::Vulkan)?;

			let scratch = GpuBuffer::create(
				loader.device,
				BufferDesc {
					size: size.build_scratch_size,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				},
			)
			.map_err(StageError::Vulkan)?;

			info.dst_acceleration_structure = as_.handle();
			info.scratch_data = vk::DeviceOrHostAddressKHR {
				device_address: scratch.addr(),
			};

			ext.cmd_build_acceleration_structures(
				loader
					.ctx
					.execute_before(QueueType::Compute)
					.map_err(StageError::Vulkan)?,
				&[info.build()],
				&[&ranges],
			);

			loader.queue.delete(scratch);
			loader.queue.delete(raw_mesh);

			as_
		};

		Ok(RRef::new(
			Mesh {
				buffer,
				submeshes,
				meshlet_count,
				acceleration_structure,
			},
			loader.deleter.clone(),
		))
	}
}

