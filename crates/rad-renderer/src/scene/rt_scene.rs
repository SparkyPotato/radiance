use std::sync::Mutex;

use ash::vk;
use bytemuck::NoUninit;
use rad_core::{
	asset::aref::{ARef, LARef},
	Engine,
};
use rad_graph::{
	device::ShaderInfo,
	graph::{BufferDesc, BufferUsage, BufferUsageType, ExternalBuffer, Frame, Res},
	resource::{ASDesc, BufferHandle, GpuPtr, Resource as _, AS},
	sync::Shader,
	util::compute::ComputePass,
};
use rad_world::{
	bevy_ecs::{
		batching::BatchingStrategy,
		component::{Component, StorageType},
		entity::Entity,
		query::{Changed, Or, Without},
		schedule::IntoSystemConfigs,
		system::{Commands, Query, ResMut, Resource},
	},
	tick::Tick,
	transform::Transform,
	TickStage,
	World,
};
use tracing::warn;

use crate::{
	assets::{
		material::GpuMaterial,
		mesh::{GpuVertex, RaytracingMeshView},
	},
	components::mesh::MeshComponent,
	scene::{should_scene_sync, GpuScene, GpuTransform},
	util::ResizableBuffer,
};

#[derive(Copy, Clone)]
pub struct RtScene {
	pub instances: Res<BufferHandle>,
	pub as_: Res<BufferHandle>,
	pub as_offset: u64,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
pub struct GpuRtInstance {
	transform: GpuTransform,
	raw_mesh: GpuPtr<GpuVertex>,
	raw_vertex_count: u32,
	raw_tri_count: u32,
	material: GpuPtr<GpuMaterial>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	instances: GpuPtr<GpuRtInstance>,
	as_instances: GpuPtr<()>,
	updates: GpuPtr<GpuRtInstanceUpdate>,
	count: u32,
	_pad: u32,
}

impl GpuScene for RtScene {
	type In = ();
	type Res = RtSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(RtSceneData::new());
		tick.add_systems(TickStage::Render, sync_rt_scene.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut RtSceneData, _: &Self::In) -> Self {
		let RtSceneData {
			update,
			instances,
			as_,
			as_instances,
			instance_count,
			updates,
		} = data;
		let count = *instance_count;

		let tinstances = instances
			.reserve(
				frame,
				"resize rt scene",
				std::mem::size_of::<GpuRtInstance>() as u64 * count as u64,
			)
			.unwrap();
		let tas_instances = as_instances
			.reserve(
				frame,
				"resize rt scene as",
				std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64 * count as u64,
			)
			.unwrap();

		let mut pass = frame.pass("update rt scene");
		let update_buf = pass.resource(
			BufferDesc::upload(std::mem::size_of::<GpuRtInstanceUpdate>() as u64 * updates.len() as u64),
			BufferUsage::read(Shader::Compute),
		);
		let instances = match tinstances {
			Some(instances) => {
				pass.reference(instances, BufferUsage::write(Shader::Compute));
				instances
			},
			None => pass.resource(
				ExternalBuffer::new(&instances.inner),
				BufferUsage::write(Shader::Compute),
			),
		};
		let as_instances_h = match tas_instances {
			Some(as_instances) => {
				pass.reference(as_instances, BufferUsage::write(Shader::Compute));
				as_instances
			},
			None => pass.resource(
				ExternalBuffer::new(&as_instances.inner),
				BufferUsage::write(Shader::Compute),
			),
		};
		pass.build(move |mut pass| {
			let count = updates.len() as u32;
			pass.write_iter(update_buf, 0, updates.drain(..));
			let instances = pass.get(instances).ptr();
			let as_instances = pass.get(as_instances_h).ptr();
			let updates = pass.get(update_buf).ptr();
			update.dispatch(
				&mut pass,
				&PushConstants {
					instances,
					as_instances,
					updates,
					count,
					_pad: 0,
				},
				(count + 63) / 64,
				1,
				1,
			);
		});

		let geo = [vk::AccelerationStructureGeometryKHR::default()
			.geometry_type(vk::GeometryTypeKHR::INSTANCES)
			.geometry(vk::AccelerationStructureGeometryDataKHR {
				instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
					.array_of_pointers(false)
					.data(vk::DeviceOrHostAddressConstKHR {
						device_address: as_instances.inner.ptr::<()>().addr(),
					}),
			})];
		let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
			.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
			.flags(
				vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
					| vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
			)
			.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
			.geometries(&geo);
		let mut sinfo = vk::AccelerationStructureBuildSizesInfoKHR::default();
		unsafe {
			frame.device().as_ext().get_acceleration_structure_build_sizes(
				vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&info,
				&[count],
				&mut sinfo,
			);
		}

		let mut curr_size = as_.size();
		if sinfo.acceleration_structure_size > curr_size {
			if curr_size == 0 {
				curr_size = 1024;
			}
			while sinfo.acceleration_structure_size > curr_size {
				curr_size *= 2;
			}
			let old = std::mem::replace(
				as_,
				AS::create(
					frame.device(),
					ASDesc {
						name: "tlas",
						flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
						ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
						size: curr_size,
					},
				)
				.unwrap(),
			);
			frame.delete(old);
		}

		let mut pass = frame.pass("build rt scene tlas");
		pass.reference(
			as_instances_h,
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildRead],
			},
		);
		let scratch = pass.resource(
			BufferDesc::gpu(sinfo.build_scratch_size),
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildScratch],
			},
		);
		let as_buf = pass.resource(
			ExternalBuffer::new(as_.inner()),
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildWrite],
			},
		);
		let dst = as_.handle();
		pass.build(move |mut pass| unsafe {
			pass.device.as_ext().cmd_build_acceleration_structures(
				pass.buf,
				&[vk::AccelerationStructureBuildGeometryInfoKHR::default()
					.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
					.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
					.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
					.geometries(&geo)
					.dst_acceleration_structure(dst)
					.scratch_data(vk::DeviceOrHostAddressKHR {
						device_address: pass.get(scratch).ptr::<u8>().addr(),
					})],
				&[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
					.primitive_count(count)
					.primitive_offset(0)]],
			);
		});

		Self {
			instances,
			as_: as_buf,
			as_offset: as_.addr() - as_.buf_handle().addr,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct GpuRtInstanceUpdate {
	index: u32,
	_pad: u32,
	as_: u64,
	instance: GpuRtInstance,
}

pub struct RtSceneData {
	update: ComputePass<PushConstants>,
	instances: ResizableBuffer,
	as_: AS,
	as_instances: ResizableBuffer,
	instance_count: u32,
	updates: Vec<GpuRtInstanceUpdate>,
}
impl Resource for RtSceneData {}

impl RtSceneData {
	pub fn new() -> Self {
		let dev = Engine::get().global();
		Self {
			update: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "asset.scene.update_rt",
					spec: &[],
				},
			)
			.unwrap(),
			instances: ResizableBuffer::new(dev, "rt scene", std::mem::size_of::<GpuRtInstance>() as u64 * 1000)
				.unwrap(),
			as_: AS::default(),
			as_instances: ResizableBuffer::new(
				dev,
				"rt as instances",
				std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64 * 1000,
			)
			.unwrap(),
			instance_count: 0,
			updates: Vec::new(),
		}
	}
}

fn map_instance(t: &Transform, m: &LARef<RaytracingMeshView>) -> (GpuRtInstance, u64) {
	(
		GpuRtInstance {
			transform: (*t).into(),
			raw_mesh: m.buffer.ptr(),
			raw_vertex_count: m.vertex_count,
			raw_tri_count: m.tri_count,
			material: m.material.gpu_ptr(),
		},
		m.as_.addr(),
	)
}

pub struct KnownRtInstances(pub Vec<(u32, LARef<RaytracingMeshView>)>);
impl Component for KnownRtInstances {
	const STORAGE_TYPE: StorageType = StorageType::Table;
}

// TODO: edits and deletion.
fn sync_rt_scene(
	mut r: ResMut<RtSceneData>, mut cmd: Commands,
	unknown: Query<(Entity, &Transform, &MeshComponent), Without<KnownRtInstances>>,
	_: Query<(&Transform, &MeshComponent, &KnownRtInstances), Or<(Changed<Transform>, Changed<MeshComponent>)>>,
) {
	let mut cache = Mutex::new(Vec::new());
	unknown
		.par_iter()
		.batching_strategy(BatchingStrategy::fixed(1))
		.for_each(|(e, t, m)| {
			let x: Vec<_> = m
				.inner
				.iter()
				.filter_map(|&m| {
					ARef::loaded(m)
						.map_err(|e| warn!("failed to load mesh {:?}: {:?}", m, e))
						.ok()
				})
				.collect();
			cache.lock().unwrap().push((e, t, x));
		});

	for (e, t, inner) in cache.into_inner().unwrap() {
		let inner = inner
			.into_iter()
			.map(|view| {
				let index = r.instance_count;
				r.instance_count += 1;
				let (instance, as_) = map_instance(t, &view);
				r.updates.push(GpuRtInstanceUpdate {
					index,
					_pad: 0,
					as_,
					instance,
				});
				(index, view)
			})
			.collect();
		cmd.entity(e).insert(KnownRtInstances(inner));
	}
}
