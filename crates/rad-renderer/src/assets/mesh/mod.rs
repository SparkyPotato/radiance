use std::{f32::consts::PI, io, usize};

use ash::vk;
use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable, cast_slice};
use rad_core::{
	Engine,
	asset::{
		AssetView,
		BincodeAsset,
		Uuid,
		aref::{ARef, AssetId, LARef},
	},
	uuid,
};
use rad_graph::{
	cmd::CommandPool,
	device::{Compute, Device, QueueWait},
	resource::{AS, ASDesc, Buffer, BufferDesc, BufferType, Resource},
	sync::{GlobalBarrier, UsageType, get_global_barrier},
};
use static_assertions::const_assert_eq;
use tracing::trace_span;
use vek::{Aabb, Mat3, Vec2, Vec3};

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

#[derive(Copy, Clone, Debug)]
pub struct NormalCone {
	pub axis: Vec3<f32>,
	pub theta_o: f32,
	pub theta_e: f32,
}

impl NormalCone {
	pub fn merge(self, other: Self) -> Self {
		if self.axis == Vec3::zero() {
			return other;
		}
		if other.axis == Vec3::zero() {
			return self;
		}

		let a = self;
		let b = other;

		let theta_e = a.theta_e.min(b.theta_e);
		let theta_d = a.axis.dot(b.axis).clamp(-1.0, 1.0).acos();
		if PI.min(theta_d + b.theta_o) <= a.theta_o {
			return Self {
				axis: a.axis,
				theta_o: a.theta_o,
				theta_e,
			};
		}
		if PI.min(theta_d + a.theta_o) <= b.theta_o {
			return Self {
				axis: b.axis,
				theta_o: b.theta_o,
				theta_e,
			};
		}
		let theta_o = (a.theta_o + b.theta_o + theta_d) / 2.0;
		if theta_o >= PI {
			return Self {
				axis: a.axis,
				theta_o: PI,
				theta_e: PI / 2.0,
			};
		}

		let theta_r = theta_o - a.theta_o;
		let w_r = a.axis.cross(b.axis);
		if w_r.magnitude_squared() < 0.00001 {
			return Self {
				axis: a.axis,
				theta_o: PI,
				theta_e: PI / 2.0,
			};
		}
		let axis = Mat3::rotation_3d(theta_r, w_r) * a.axis;

		Self { axis, theta_o, theta_e }
	}
}

pub struct RaytracingMeshView {
	pub buffer: Buffer,
	pub as_: AS,
	pub vertex_count: u32,
	pub tri_count: u32,
	pub material: LARef<MaterialView>,
	pub aabb: Aabb<f32>,
	pub normal_average: Vec3<f32>,
	pub normal_cone: NormalCone,
	pub area: f32,
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
					ty: BufferType::Gpu,
				},
			)?;
			let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });
			writer.write_slice(&m.vertices);
			writer.write_slice(&m.indices);
			buffer
		};

		let tri_count = m.indices.len() as u32 / 3;
		unsafe {
			let mut pool = CommandPool::new(device, device.queue_families().into::<Compute>())?;
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
						ty: BufferType::Gpu,
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
				let sync = device.submit::<Compute>(QueueWait::default(), &[cmd], &[])?;
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
				let sync = device.submit::<Compute>(QueueWait::default(), &[cmd], &[])?;
				sync.wait(device)?;
				pool.destroy(device);
				old.destroy(device);
				as_
			};

			let (aabb, norm_sum) = m.vertices.iter().map(|x| (x.position, x.normal)).fold(
				(
					Aabb {
						min: Vec3::broadcast(f32::INFINITY),
						max: Vec3::broadcast(f32::NEG_INFINITY),
					},
					Vec3::zero(),
				),
				|a, b| {
					(
						Aabb {
							min: Vec3::partial_min(a.0.min, b.0),
							max: Vec3::partial_max(a.0.max, b.0),
						},
						a.1 + b.1,
					)
				},
			);
			let (normal_cone, area) = m.indices.chunks_exact(3).fold(
				(
					NormalCone {
						axis: Vec3::zero(),
						theta_o: 0.0,
						theta_e: 0.0,
					},
					0.0,
				),
				|(cone, area), tri| {
					let v1 = m.vertices[tri[0] as usize].position;
					let v2 = m.vertices[tri[1] as usize].position;
					let v3 = m.vertices[tri[2] as usize].position;
					let normal = (v2 - v1).cross(v3 - v1);
					let len = normal.magnitude();
					let area = area + len / 2.0;
					let c = NormalCone {
						axis: if len < 0.0001 { Vec3::zero() } else { normal / len },
						theta_o: 0.0,
						theta_e: PI / 2.0,
					};
					(cone.merge(c), area)
				},
			);

			Ok(Self {
				buffer,
				as_,
				vertex_count: m.vertices.len() as _,
				tri_count,
				material: ARef::loaded(m.material)?,
				aabb,
				normal_cone,
				normal_average: norm_sum / m.vertices.len() as f32,
				area,
			})
		}
	}
}
