use std::{io, usize};

use ash::vk;
use bincode::{Decode, Encode};
use bytemuck::{cast_slice, Pod, Zeroable};
use rad_core::{
	asset::{
		aref::{ARef, AssetId, LARef},
		AssetView,
		BincodeAsset,
		Uuid,
	},
	uuid,
	Engine,
};
use rad_graph::{
	cmd::CommandPool,
	device::{Compute, Device, QueueWait},
	resource::{ASDesc, Buffer, BufferDesc, Resource, AS},
	sync::{get_global_barrier, GlobalBarrier, UsageType},
};
use static_assertions::const_assert_eq;
use tracing::trace_span;
use vek::{Vec2, Vec3};

use crate::{
	assets::material::{Material, MaterialView},
	util::SliceWriter,
};

pub mod virtual_mesh;

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
pub type GpuVertex = Vertex;

const_assert_eq!(std::mem::size_of::<Vertex>(), 32);
const_assert_eq!(std::mem::align_of::<Vertex>(), 4);

#[derive(Encode, Decode)]
pub struct Mesh {
	pub vertices: Vec<Vertex>,
	pub indices: Vec<u32>,
	pub material: AssetId<Material>,
}

impl BincodeAsset for Mesh {
	const UUID: Uuid = uuid!("63d17036-5d82-4d70-a15e-103e72559abe");
}

pub struct RaytracingMeshView {
	pub buffer: Buffer,
	pub as_: AS,
	pub vertex_count: u32,
	pub tri_count: u32,
	pub material: LARef<MaterialView>,
}

impl AssetView for RaytracingMeshView {
	type Base = Mesh;
	type Ctx = ();

	fn load(_: &'static Self::Ctx, m: Self::Base) -> Result<Self, io::Error> {
		let device: &Device = Engine::get().global();
		// TODO: fips.
		let name = "raytracing mesh";
		let s = trace_span!("load raytracing mesh", name = name);
		let _e = s.enter();

		let buffer = {
			let s = trace_span!("load");
			let _e = s.enter();
			let buffer = Buffer::create(
				device,
				BufferDesc {
					name: &format!("{name} raw buffer"),
					size: (cast_slice::<_, u8>(&m.vertices).len() + cast_slice::<_, u8>(&m.indices).len()) as u64,
					readback: false,
				},
			)?;
			let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });
			writer.write_slice(&m.vertices);
			writer.write_slice(&m.indices);
			buffer
		};

		let tri_count = m.indices.len() as u32 / 3;
		unsafe {
			let mut pool = CommandPool::new(device, device.queue_families().compute)?;
			let qpool = device
				.device()
				.create_query_pool(
					&vk::QueryPoolCreateInfo::default()
						.query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
						.query_count(1),
					None,
				)
				.unwrap();
			let cmd = pool.next(device)?;
			let old = {
				let s = trace_span!("build AS");
				let _e = s.enter();
				let geo = [vk::AccelerationStructureGeometryKHR::default()
					.geometry_type(vk::GeometryTypeKHR::TRIANGLES)
					.geometry(vk::AccelerationStructureGeometryDataKHR {
						triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
							.vertex_format(vk::Format::R32G32B32_SFLOAT)
							.vertex_data(vk::DeviceOrHostAddressConstKHR {
								device_address: buffer.ptr::<u8>().addr(),
							})
							.vertex_stride(std::mem::size_of::<GpuVertex>() as _)
							.max_vertex(m.indices.len() as u32 - 1)
							.index_type(vk::IndexType::UINT32)
							.index_data(vk::DeviceOrHostAddressConstKHR {
								device_address: buffer.ptr::<u8>().addr()
									+ cast_slice::<_, u8>(&m.vertices).len() as u64,
							}),
					})];
				let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
					.ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
					.flags(
						vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
							| vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION,
					)
					.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
					.geometries(&geo);
				let mut sinfo = vk::AccelerationStructureBuildSizesInfoKHR::default();
				device.as_ext().get_acceleration_structure_build_sizes(
					vk::AccelerationStructureBuildTypeKHR::DEVICE,
					&info,
					&[tri_count],
					&mut sinfo,
				);

				let old = AS::create(
					device,
					ASDesc {
						name: &format!("{name} uncompacted AS"),
						flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
						ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
						size: sinfo.acceleration_structure_size,
					},
				)?;
				let scratch = Buffer::create(
					device,
					BufferDesc {
						name: &format!("{name} AS build scratch"),
						size: sinfo.build_scratch_size,
						readback: false,
					},
				)?;
				info.dst_acceleration_structure = old.handle();
				info.scratch_data.device_address = scratch.ptr::<u8>().addr();

				device
					.device()
					.begin_command_buffer(
						cmd,
						&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
					)
					.unwrap();
				device.device().cmd_reset_query_pool(cmd, qpool, 0, 1);
				device.as_ext().cmd_build_acceleration_structures(
					cmd,
					&[info],
					&[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
						.primitive_count(tri_count)
						.primitive_offset(0)
						.first_vertex(0)]],
				);
				device.device().cmd_pipeline_barrier2(
					cmd,
					&vk::DependencyInfo::default().memory_barriers(&[get_global_barrier(&GlobalBarrier {
						previous_usages: &[UsageType::AccelerationStructureBuildWrite],
						next_usages: &[UsageType::AccelerationStructureBuildRead],
					})]),
				);
				device.as_ext().cmd_write_acceleration_structures_properties(
					cmd,
					&[old.handle()],
					vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
					qpool,
					0,
				);
				device.device().end_command_buffer(cmd).unwrap();
				let sync = device.submit::<Compute>(QueueWait::default(), &[cmd], &[], vk::Fence::null())?;
				sync.wait(device)?;
				pool.reset(device)?;
				scratch.destroy(device);

				old
			};

			let as_ = {
				let s = trace_span!("compact AS");
				let _e = s.enter();
				let mut size = [0u64];
				device
					.device()
					.get_query_pool_results(
						qpool,
						0,
						&mut size,
						vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
					)
					.unwrap();
				device.device().destroy_query_pool(qpool, None);

				let as_ = AS::create(
					device,
					ASDesc {
						name: &format!("{name} AS"),
						flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
						ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
						size: size[0],
					},
				)?;
				device
					.device()
					.begin_command_buffer(
						cmd,
						&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
					)
					.unwrap();
				device.as_ext().cmd_copy_acceleration_structure(
					cmd,
					&vk::CopyAccelerationStructureInfoKHR::default()
						.src(old.handle())
						.dst(as_.handle())
						.mode(vk::CopyAccelerationStructureModeKHR::COMPACT),
				);
				device.device().end_command_buffer(cmd).unwrap();
				let sync = device.submit::<Compute>(QueueWait::default(), &[cmd], &[], vk::Fence::null())?;
				sync.wait(device)?;
				pool.destroy(device);
				old.destroy(device);
				as_
			};

			Ok(Self {
				buffer,
				as_,
				vertex_count: m.vertices.len() as _,
				tri_count,
				material: ARef::loaded(m.material)?,
			})
		}
	}
}
