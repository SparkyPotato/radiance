use std::sync::Mutex;

use bytemuck::NoUninit;
use rad_core::{
	asset::aref::{ARef, LARef},
	Engine,
};
use rad_graph::{
	device::ShaderInfo,
	graph::{BufferDesc, BufferUsage, ExternalBuffer, Frame, Res},
	resource::{BufferHandle, GpuPtr},
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
		mesh::virtual_mesh::{GpuAabb, VirtualMeshView},
	},
	components::mesh::MeshComponent,
	scene::{should_scene_sync, GpuScene, GpuTransform},
	util::ResizableBuffer,
};

#[derive(Copy, Clone)]
pub struct VirtualScene {
	pub instances: Res<BufferHandle>,
	pub instance_count: u32,
	pub bvh_depth: u32,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
pub struct GpuInstance {
	transform: GpuTransform,
	last_updated_transform: GpuTransform,
	aabb: GpuAabb,
	last_updated_frame: u64,
	mesh: GpuPtr<u8>,
	material: GpuPtr<GpuMaterial>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	updates: GpuPtr<GpuInstanceUpdate>,
	count: u32,
	_pad: u32,
}

impl GpuScene for VirtualScene {
	type In = ();
	type Res = VirtualSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(VirtualSceneData::new());
		tick.add_systems(TickStage::Render, sync_virtual_scene.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut VirtualSceneData, _: &Self::In) -> Self {
		let VirtualSceneData {
			update,
			instances,
			instance_count,
			bvh_depth,
			updates,
		} = data;
		let instance_count = *instance_count;
		let bvh_depth = *bvh_depth;

		let tinstances = instances
			.reserve(
				frame,
				"resize virtual scene",
				std::mem::size_of::<GpuInstance>() as u64 * instance_count as u64,
			)
			.unwrap();

		let mut pass = frame.pass("update virtual scene");
		let update_buf = pass.resource(
			BufferDesc::upload(std::mem::size_of::<GpuInstanceUpdate>() as u64 * updates.len() as u64),
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
		pass.build(move |mut pass| {
			let count = updates.len() as u32;
			pass.write_iter(update_buf, 0, updates.drain(..));
			let instances = pass.get(instances).ptr();
			let updates = pass.get(update_buf).ptr();
			update.dispatch(
				&mut pass,
				&PushConstants {
					instances,
					updates,
					count,
					_pad: 0,
				},
				(count + 63) / 64,
				1,
				1,
			);
		});

		Self {
			instances,
			instance_count,
			bvh_depth,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct GpuInstanceUpdate {
	index: u32,
	_pad: u32,
	instance: GpuInstance,
}

pub struct VirtualSceneData {
	update: ComputePass<PushConstants>,
	instances: ResizableBuffer,
	instance_count: u32,
	bvh_depth: u32,
	updates: Vec<GpuInstanceUpdate>,
}
impl Resource for VirtualSceneData {}

impl VirtualSceneData {
	fn new() -> Self {
		let dev = Engine::get().global();
		Self {
			update: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "asset.scene.update_virtual",
					spec: &[],
				},
			)
			.unwrap(),
			instances: ResizableBuffer::new(dev, "virtual scene", std::mem::size_of::<GpuInstance>() as u64 * 1000)
				.unwrap(),
			instance_count: 0,
			bvh_depth: 0,
			updates: Vec::new(),
		}
	}

	fn push_instance(&mut self, index: u32, t: &Transform, m: &LARef<VirtualMeshView>) {
		self.updates.push(GpuInstanceUpdate {
			index,
			_pad: 0,
			instance: GpuInstance {
				transform: (*t).into(),
				last_updated_transform: (*t).into(),
				aabb: m.gpu_aabb(),
				last_updated_frame: 0,
				mesh: m.gpu_ptr(),
				material: m.material().gpu_ptr(),
			},
		});
		self.bvh_depth = self.bvh_depth.max(m.bvh_depth());
	}
}

pub struct KnownVirtualInstances(pub Vec<(u32, LARef<VirtualMeshView>)>);
impl Component for KnownVirtualInstances {
	const STORAGE_TYPE: StorageType = StorageType::Table;
}

// TODO: edits and deletion.
fn sync_virtual_scene(
	mut r: ResMut<VirtualSceneData>, mut cmd: Commands,
	unknown: Query<(Entity, &Transform, &MeshComponent), Without<KnownVirtualInstances>>,
	_: Query<(&Transform, &MeshComponent, &KnownVirtualInstances), Or<(Changed<Transform>, Changed<MeshComponent>)>>,
) {
	let cache = Mutex::new(Vec::new());
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
				r.push_instance(index, t, &view);
				(index, view)
			})
			.collect();
		cmd.entity(e).insert(KnownVirtualInstances(inner));
	}
}
