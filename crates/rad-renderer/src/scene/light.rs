use bytemuck::NoUninit;
use rad_core::{Engine, asset::aref::LARef};
use rad_graph::{
	device::ShaderInfo,
	graph::{BufferDesc, BufferUsage, ExternalBuffer, Frame},
	resource::{self, Buffer, BufferType, GpuPtr, Resource as _},
	sync::Shader,
	util::compute::ComputePass,
};
use rad_world::{
	TickStage,
	World,
	bevy_ecs::{
		component::{Component, StorageType},
		entity::Entity,
		query::Without,
		schedule::IntoSystemConfigs,
		system::{Commands, Query, ResMut, Resource},
	},
	tick::Tick,
	transform::Transform,
};
use rustc_hash::FxHashMap;
use vek::Vec3;

use crate::{
	assets::mesh::{RaytracingMeshView, Vertex},
	components::light::LightComponent,
	scene::{GpuScene, rt_scene::KnownRtInstances, should_scene_sync},
	sort::GpuSorter,
};

#[derive(Copy, Clone)]
pub struct LightScene {}

impl GpuScene for LightScene {
	type In = ();
	type Res = LightSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(LightSceneData::new());
		tick.add_systems(TickStage::Render, sync_lights.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut LightSceneData, _: &Self::In) -> Self {
		for mesh in data.mesh_build_queue.drain(..) {
			let n = mesh.tri_count;
			let buffer = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "mesh light tree",
					size: (32 * (n - 1)) as _,
					ty: BufferType::Gpu,
				},
			)
			.unwrap();
			let handle = buffer.handle();
			data.mesh_bvhs.insert(mesh.clone(), buffer);

			frame.start_region("build mesh light tree");

			let mut push = BvhPushConstants {
				bvh_nodes: handle.ptr(),
				atomic: GpuPtr::null(),
				cluster_indices: GpuPtr::null(),
				codes: GpuPtr::null(),
				parent_ids: GpuPtr::null(),
				vertices: mesh.vertices(),
				indices: mesh.indices(),
				root_bounds: [mesh.aabb.min, mesh.aabb.max],
				tri_count: n,
				_pad: 0,
			};

			let mut pass = frame.pass("sfc");
			let mut buf = |size: u64| {
				pass.resource(
					BufferDesc::gpu(size * std::mem::size_of::<u32>() as u64),
					BufferUsage::write(Shader::Compute),
				)
			};
			let atomic = buf(1);
			let cluster_indices = buf(n as u64);
			let codes = buf(n as u64 * 2);
			let parent_ids = buf(n as u64);
			let sfc = &data.sfc;
			pass.build(move |mut pass| {
				push.atomic = pass.get(atomic).ptr();
				push.cluster_indices = pass.get(cluster_indices).ptr();
				push.codes = pass.get(codes).ptr();
				push.parent_ids = pass.get(parent_ids).ptr();
				sfc.dispatch(&mut pass, &push, n.div_ceil(128), 1, 1);
			});

			let (codes, cluster_indices) = data.sorter.sort(frame, codes, cluster_indices, n);

			let mut pass = frame.pass("build");
			pass.resource(ExternalBuffer { handle }, BufferUsage::write(Shader::Compute));
			pass.reference(atomic, BufferUsage::read_write(Shader::Compute));
			pass.reference(cluster_indices, BufferUsage::read_write(Shader::Compute));
			pass.reference(codes, BufferUsage::read(Shader::Compute));
			pass.reference(parent_ids, BufferUsage::read_write(Shader::Compute));
			let build = &data.build;
			pass.build(move |mut pass| {
				push.atomic = pass.get(atomic).ptr();
				push.cluster_indices = pass.get(cluster_indices).ptr();
				push.codes = pass.get(codes).ptr();
				push.parent_ids = pass.get(parent_ids).ptr();
				build.dispatch(&mut pass, &push, n.div_ceil(32), 1, 1);
			});

			frame.end_region();
		}

		Self {}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(u32)]
pub enum GpuLightType {
	Point,
	Directional,
	Emissive,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuLight {
	pub ty: GpuLightType,
	pub radiance: Vec3<f32>,
	pub pos_or_dir: Vec3<f32>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct GpuLightUpdate {
	index: u32,
	light: GpuLight,
}

// TODO: global the pipeline.
pub struct LightSceneData {
	sfc: ComputePass<BvhPushConstants>,
	build: ComputePass<BvhPushConstants>,
	sorter: GpuSorter,
	mesh_bvhs: FxHashMap<LARef<RaytracingMeshView>, Buffer>,
	mesh_build_queue: Vec<LARef<RaytracingMeshView>>,
}
impl Resource for LightSceneData {}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct BvhPushConstants {
	bvh_nodes: GpuPtr<()>,
	atomic: GpuPtr<u32>,
	cluster_indices: GpuPtr<u32>,
	codes: GpuPtr<u64>,
	parent_ids: GpuPtr<u32>,
	vertices: GpuPtr<Vertex>,
	indices: GpuPtr<u32>,
	root_bounds: [Vec3<f32>; 2],
	tri_count: u32,
	_pad: u32,
}

impl LightSceneData {
	fn new() -> Self {
		let dev = Engine::get().global();
		Self {
			sfc: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "scene.light_bvh.hploc_sfc",
					spec: &[],
				},
			)
			.unwrap(),
			build: ComputePass::with_wave_32(
				dev,
				ShaderInfo {
					shader: "scene.light_bvh.hploc_build",
					spec: &[],
				},
			)
			.unwrap(),
			sorter: GpuSorter::new(dev).unwrap(),
			mesh_bvhs: FxHashMap::default(),
			mesh_build_queue: Vec::new(),
		}
	}
}

struct KnownLight;
impl Component for KnownLight {
	const STORAGE_TYPE: StorageType = StorageType::Table;
}

// TODO: figure out how deal with component or entity removal.
fn sync_lights(
	mut r: ResMut<LightSceneData>, mut cmd: Commands,
	unknown_punctual: Query<(Entity, &Transform, &LightComponent), Without<KnownLight>>,
	unknown_emissive: Query<(Entity, &KnownRtInstances), Without<KnownLight>>,
) {
	for (e, m) in unknown_emissive.iter() {
		for (_, mesh) in m.0.iter() {
			if mesh.material.emissive_factor == Vec3::zero() {
				continue;
			}

			r.mesh_build_queue.push(mesh.clone());
		}
		cmd.entity(e).insert(KnownLight);
	}
}
