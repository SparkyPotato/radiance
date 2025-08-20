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
use vek::{Aabb, Vec3};

use crate::{
	assets::{
		material::GpuMaterial,
		mesh::{RaytracingMeshView, Vertex, virtual_mesh::aabb_default},
	},
	components::light::LightComponent,
	scene::{GpuScene, GpuTransform, rt_scene::KnownRtInstances, should_scene_sync},
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
					size: (48 * (n - 1)) as _,
					ty: BufferType::Gpu,
				},
			)
			.unwrap();
			let root = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "mesh light tree root",
					size: 112,
					ty: BufferType::Gpu,
				},
			)
			.unwrap();

			let handle = buffer.handle();
			let root_handle = root.handle();
			let mut push = BvhPushConstants {
				build: Blas {
					nodes: handle.ptr(),
					vertices: mesh.vertices(),
					indices: mesh.indices(),
					material: mesh.material.gpu_ptr(),
					root: root_handle.ptr(),
				},
				bvh_nodes: GpuPtr::null(),
				atomic: GpuPtr::null(),
				cluster_indices: GpuPtr::null(),
				codes: GpuPtr::null(),
				parent_ids: GpuPtr::null(),
				root_bounds: [mesh.aabb.min, mesh.aabb.max],
				prim_count: n,
				_pad: 0,
			};
			data.mesh_bvhs.insert(mesh, (buffer, root));

			frame.start_region("build mesh light tree");

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
			let sfc = &data.blas_sfc;
			pass.build(move |mut pass| {
				push.atomic = pass.get(atomic).ptr();
				push.cluster_indices = pass.get(cluster_indices).ptr();
				push.codes = pass.get(codes).ptr();
				push.parent_ids = pass.get(parent_ids).ptr();
				sfc.dispatch(&mut pass, &push, n.div_ceil(128), 1, 1);
			});

			let (codes, cluster_indices) = data.sorter.sort(frame, codes, cluster_indices, n);

			let mut pass = frame.pass("build");
			let bvh_nodes = pass.resource(
				BufferDesc::gpu((88 * (n - 1)) as _),
				BufferUsage::read_write(Shader::Compute),
			);
			pass.resource(ExternalBuffer { handle }, BufferUsage::read_write(Shader::Compute));
			pass.resource(
				ExternalBuffer { handle: root_handle },
				BufferUsage::write(Shader::Compute),
			);
			pass.reference(atomic, BufferUsage::read_write(Shader::Compute));
			pass.reference(cluster_indices, BufferUsage::read_write(Shader::Compute));
			pass.reference(codes, BufferUsage::read(Shader::Compute));
			pass.reference(parent_ids, BufferUsage::read_write(Shader::Compute));
			let build = &data.blas_build;
			pass.build(move |mut pass| {
				push.bvh_nodes = pass.get(bvh_nodes).ptr();
				push.atomic = pass.get(atomic).ptr();
				push.cluster_indices = pass.get(cluster_indices).ptr();
				push.codes = pass.get(codes).ptr();
				push.parent_ids = pass.get(parent_ids).ptr();
				build.dispatch(&mut pass, &push, n.div_ceil(32), 1, 1);
			});

			frame.end_region();
		}

		let n = data.meshes.len() as u32;
		if n == 0 {
			return Self {};
		}

		let mut push = BvhPushConstants {
			build: Tlas {
				nodes: GpuPtr::null(),
				roots: GpuPtr::null(),
			},
			bvh_nodes: GpuPtr::null(),
			atomic: GpuPtr::null(),
			cluster_indices: GpuPtr::null(),
			codes: GpuPtr::null(),
			parent_ids: GpuPtr::null(),
			root_bounds: [data.scene_aabb.min, data.scene_aabb.max],
			prim_count: n,
			_pad: 0,
		};

		frame.start_region("build scene light tree");

		let mut pass = frame.pass("sfc");
		let roots = pass.resource(
			BufferDesc::upload((std::mem::size_of::<BlasInstance>() as u32 * n) as _),
			BufferUsage::read(Shader::Compute),
		);
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
		let sfc = &data.tlas_sfc;
		let bvhs = &data.mesh_bvhs;
		let meshes = &mut data.meshes;
		pass.build(move |mut pass| {
			pass.write_iter(
				roots,
				0,
				meshes.iter().map(|(r, t)| BlasInstance {
					transform: *t,
					blas: bvhs[r].1.ptr(),
				}),
			);
			push.build.roots = pass.get(roots).ptr();
			push.atomic = pass.get(atomic).ptr();
			push.cluster_indices = pass.get(cluster_indices).ptr();
			push.codes = pass.get(codes).ptr();
			push.parent_ids = pass.get(parent_ids).ptr();
			sfc.dispatch(&mut pass, &push, n.div_ceil(128), 1, 1);
		});

		let (codes, cluster_indices) = data.sorter.sort(frame, codes, cluster_indices, n);

		let mut pass = frame.pass("build");
		let tree_nodes = pass.resource(
			BufferDesc::gpu((48 * (n - 1)) as _),
			BufferUsage::read_write(Shader::Compute),
		);
		let bvh_nodes = pass.resource(
			BufferDesc::gpu((88 * (n - 1)) as _),
			BufferUsage::read_write(Shader::Compute),
		);
		pass.reference(roots, BufferUsage::read(Shader::Compute));
		pass.reference(atomic, BufferUsage::read_write(Shader::Compute));
		pass.reference(cluster_indices, BufferUsage::read_write(Shader::Compute));
		pass.reference(codes, BufferUsage::read(Shader::Compute));
		pass.reference(parent_ids, BufferUsage::read_write(Shader::Compute));
		let build = &data.tlas_build;
		pass.build(move |mut pass| {
			push.build.nodes = pass.get(tree_nodes).ptr();
			push.build.roots = pass.get(roots).ptr();
			push.bvh_nodes = pass.get(bvh_nodes).ptr();
			push.atomic = pass.get(atomic).ptr();
			push.cluster_indices = pass.get(cluster_indices).ptr();
			push.codes = pass.get(codes).ptr();
			push.parent_ids = pass.get(parent_ids).ptr();
			build.dispatch(&mut pass, &push, n.div_ceil(32), 1, 1);
		});

		frame.end_region();

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
	blas_sfc: ComputePass<BvhPushConstants<Blas>>,
	blas_build: ComputePass<BvhPushConstants<Blas>>,
	tlas_sfc: ComputePass<BvhPushConstants<Tlas>>,
	tlas_build: ComputePass<BvhPushConstants<Tlas>>,
	sorter: GpuSorter,
	mesh_bvhs: FxHashMap<LARef<RaytracingMeshView>, (Buffer, Buffer)>,
	meshes: Vec<(LARef<RaytracingMeshView>, GpuTransform)>,
	mesh_build_queue: Vec<LARef<RaytracingMeshView>>,
	scene_aabb: Aabb<f32>,
}
impl Resource for LightSceneData {}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct Blas {
	nodes: GpuPtr<()>,
	vertices: GpuPtr<Vertex>,
	indices: GpuPtr<u32>,
	material: GpuPtr<GpuMaterial>,
	root: GpuPtr<()>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct BlasInstance {
	transform: GpuTransform,
	blas: GpuPtr<()>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct Tlas {
	nodes: GpuPtr<()>,
	roots: GpuPtr<BlasInstance>,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct BvhPushConstants<Ty> {
	build: Ty,
	bvh_nodes: GpuPtr<()>,
	atomic: GpuPtr<u32>,
	cluster_indices: GpuPtr<u32>,
	codes: GpuPtr<u64>,
	parent_ids: GpuPtr<u32>,
	root_bounds: [Vec3<f32>; 2],
	prim_count: u32,
	_pad: u32,
}
unsafe impl<Ty: NoUninit> NoUninit for BvhPushConstants<Ty> {}

impl LightSceneData {
	fn new() -> Self {
		let dev = Engine::get().global();
		Self {
			blas_sfc: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "scene.light_tree_build.hploc_sfc",
					spec: &["scene.light_tree_build_blas"],
				},
			)
			.unwrap(),
			blas_build: ComputePass::with_wave_32(
				dev,
				ShaderInfo {
					shader: "scene.light_tree_build.hploc_build",
					spec: &["scene.light_tree_build_blas"],
				},
			)
			.unwrap(),
			tlas_sfc: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "scene.light_tree_build.hploc_sfc",
					spec: &["scene.light_tree_build_tlas"],
				},
			)
			.unwrap(),
			tlas_build: ComputePass::with_wave_32(
				dev,
				ShaderInfo {
					shader: "scene.light_tree_build.hploc_build",
					spec: &["scene.light_tree_build_tlas"],
				},
			)
			.unwrap(),
			sorter: GpuSorter::new(dev).unwrap(),
			mesh_bvhs: FxHashMap::default(),
			meshes: Vec::new(),
			mesh_build_queue: Vec::new(),
			scene_aabb: aabb_default(),
		}
	}
}

struct KnownLight;
impl Component for KnownLight {
	const STORAGE_TYPE: StorageType = StorageType::Table;
}

fn transform_aabb(a: Aabb<f32>, transform: &Transform) -> Aabb<f32> {
	let mat = transform.into_matrix();
	let corners = [
		Vec3::new(a.min.x, a.min.y, a.min.z),
		Vec3::new(a.max.x, a.min.y, a.min.z),
		Vec3::new(a.min.x, a.max.y, a.min.z),
		Vec3::new(a.max.x, a.max.y, a.min.z),
		Vec3::new(a.min.x, a.min.y, a.max.z),
		Vec3::new(a.max.x, a.min.y, a.max.z),
		Vec3::new(a.min.x, a.max.y, a.max.z),
		Vec3::new(a.max.x, a.max.y, a.max.z),
	];
	let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
	let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
	for c in corners {
		let c = (mat * c.with_w(1.0)).xyz();
		min = Vec3::partial_min(min, c);
		max = Vec3::partial_max(max, c);
	}
	Aabb { min, max }
}

// TODO: figure out how deal with component or entity removal.
fn sync_lights(
	mut r: ResMut<LightSceneData>, mut cmd: Commands,
	unknown_punctual: Query<(Entity, &Transform, &LightComponent), Without<KnownLight>>,
	unknown_emissive: Query<(Entity, &Transform, &KnownRtInstances), Without<KnownLight>>,
) {
	for (e, t, m) in unknown_emissive.iter() {
		for (_, mesh) in m.0.iter() {
			if mesh.material.emissive_factor == Vec3::zero() {
				continue;
			}

			let aabb = transform_aabb(mesh.aabb, t);
			r.scene_aabb.min = Vec3::partial_min(r.scene_aabb.min, aabb.min);
			r.scene_aabb.max = Vec3::partial_max(r.scene_aabb.max, aabb.max);
			r.mesh_build_queue.push(mesh.clone());
			r.meshes.push((mesh.clone(), (*t).into()));
		}
		cmd.entity(e).insert(KnownLight);
	}
}
