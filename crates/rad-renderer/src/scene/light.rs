use bytemuck::NoUninit;
use rad_core::Engine;
use rad_graph::{
	device::ShaderInfo,
	graph::{BufferDesc, BufferUsage, ExternalBuffer, Frame, Res},
	resource::{BufferHandle, GpuPtr},
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
use vek::Vec3;

use crate::{
	components::light::{LightComponent, LightType},
	scene::{GpuScene, rt_scene::KnownRtInstances, should_scene_sync},
	util::ResizableBuffer,
};

#[derive(Copy, Clone)]
pub struct LightScene {
	pub buf: Res<BufferHandle>,
	pub count: u32,
	pub sun_radiance: Vec3<f32>,
	pub sun_dir: Vec3<f32>,
}

impl GpuScene for LightScene {
	type In = ();
	type Res = LightSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(LightSceneData::new());
		tick.add_systems(TickStage::Render, sync_lights.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut LightSceneData, _: &Self::In) -> Self {
		let buf = data
			.buf
			.reserve(
				frame,
				"resize light scene",
				std::mem::size_of::<GpuLight>() as u64 * data.light_count as u64,
			)
			.unwrap();

		let mut pass = frame.pass("update light scene");
		let updates = pass.resource(
			BufferDesc::upload(std::mem::size_of::<GpuLightUpdate>() as u64 * data.updates.len() as u64),
			BufferUsage::read(Shader::Compute),
		);
		let buf = match buf {
			Some(buf) => {
				pass.reference(buf, BufferUsage::write(Shader::Compute));
				buf
			},
			None => pass.resource(
				ExternalBuffer::new(&data.buf.inner),
				BufferUsage::write(Shader::Compute),
			),
		};
		let count = data.light_count;
		let sun_radiance = data.sun_radiance;
		let sun_dir = data.sun_dir;
		pass.build(move |mut pass| {
			let count = data.updates.len() as u32;
			pass.write_iter(updates, 0, data.updates.drain(..));
			let lights = pass.get(buf).ptr();
			let updates = pass.get(updates).ptr();
			data.update.dispatch(
				&mut pass,
				&PushConstants {
					lights,
					updates,
					count,
					_pad: 0,
				},
				count.div_ceil(64),
				1,
				1,
			);
		});
		Self {
			buf,
			count,
			sun_radiance,
			sun_dir,
		}
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
	update: ComputePass<PushConstants>,
	buf: ResizableBuffer,
	updates: Vec<GpuLightUpdate>,
	light_count: u32,
	sun_radiance: Vec3<f32>,
	sun_dir: Vec3<f32>,
}
impl Resource for LightSceneData {}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	lights: GpuPtr<GpuLight>,
	updates: GpuPtr<GpuLightUpdate>,
	count: u32,
	_pad: u32,
}

impl LightSceneData {
	fn new() -> Self {
		let dev = Engine::get().global();
		Self {
			update: ComputePass::new(
				dev,
				ShaderInfo {
					shader: "asset.scene.update_light",
					spec: &[],
				},
			)
			.unwrap(),
			buf: ResizableBuffer::new(dev, "light scene", std::mem::size_of::<GpuLight>() as u64 * 1000).unwrap(),
			updates: Vec::new(),
			light_count: 0,
			sun_radiance: Vec3::zero(),
			sun_dir: -Vec3::unit_z(),
		}
	}

	fn push_light(&mut self, index: u32, t: &Transform, l: &LightComponent) {
		self.updates.push(GpuLightUpdate {
			index,
			light: GpuLight {
				ty: match l.ty {
					LightType::Point => GpuLightType::Point,
					LightType::Directional => GpuLightType::Directional,
				},
				radiance: l.radiance,
				pos_or_dir: match l.ty {
					LightType::Point => t.position,
					LightType::Directional => t.rotation * -Vec3::unit_z(),
				},
			},
		});

		if matches!(l.ty, LightType::Directional) {
			self.sun_radiance = l.radiance;
			self.sun_dir = t.rotation * -Vec3::unit_z();
		}
	}

	fn push_emissive(&mut self, index: u32, mesh_index: u32) {
		self.updates.push(GpuLightUpdate {
			index,
			light: GpuLight {
				ty: GpuLightType::Emissive,
				radiance: Vec3::new(f32::from_bits(mesh_index), 0.0, 0.0),
				pos_or_dir: Vec3::zero(),
			},
		});
	}
}

struct KnownLight(Vec<u32>);
impl Component for KnownLight {
	const STORAGE_TYPE: StorageType = StorageType::Table;
}

// TODO: figure out how deal with component or entity removal.
fn sync_lights(
	mut r: ResMut<LightSceneData>, mut cmd: Commands,
	unknown_punctual: Query<(Entity, &Transform, &LightComponent), Without<KnownLight>>,
	unknown_emissive: Query<(Entity, &KnownRtInstances), Without<KnownLight>>,
) {
	for (e, t, l) in unknown_punctual.iter() {
		let index = r.light_count;
		r.light_count += 1;
		r.push_light(index, t, l);
		cmd.entity(e).insert(KnownLight(vec![index]));
	}
	for (e, m) in unknown_emissive.iter() {
		let m = m.0.iter().map(|(i, v)| (i, &v.material));
		let mut inner = Vec::new();
		for (&i, m) in m {
			if m.emissive_factor == Vec3::zero() {
				continue;
			}

			let index = r.light_count;
			r.light_count += 1;
			r.push_emissive(index, i);
			inner.push(index);
		}
		cmd.entity(e).insert(KnownLight(inner));
	}
}
