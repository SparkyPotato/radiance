use bytemuck::NoUninit;
use rad_graph::{
	graph::{BufferDesc, BufferUsage, Frame, Res},
	resource::BufferHandle,
};
use rad_world::{
	TickStage,
	World,
	bevy_ecs::{
		query::With,
		schedule::IntoSystemConfigs,
		system::{Query, ResMut, Resource},
	},
	tick::Tick,
	transform::Transform,
};
use tracing::warn;

use crate::{
	components::camera::{CameraComponent, PrimaryViewComponent},
	scene::{GpuScene, GpuTransform, should_scene_sync},
};

#[repr(C)]
#[derive(Copy, Clone, Default, NoUninit)]
pub struct GpuCamera {
	transform: GpuTransform,
	w: f32,
	h: f32,
	near: f32,
}

impl GpuCamera {
	pub fn new(aspect: f32, camera: Camera) -> Self {
		let h = (camera.camera.fov / 2.0).tan().recip();
		let w = h / aspect;
		Self {
			transform: camera.transform.into(),
			w,
			h,
			near: camera.camera.near,
		}
	}
}

#[derive(Copy, Clone)]
pub struct CameraScene {
	pub buf: Res<BufferHandle>,
	pub prev: Camera,
	pub curr: Camera,
}

pub struct CameraSceneInfo {
	pub aspect: f32,
}

impl GpuScene for CameraScene {
	type In = CameraSceneInfo;
	type Res = CameraSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(CameraSceneData::default());
		tick.add_systems(TickStage::Render, find_primary_view.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut CameraSceneData, input: &Self::In) -> Self {
		let mut pass = frame.pass("update camera scene");
		let buf = pass.resource(
			BufferDesc::upload(std::mem::size_of::<[GpuCamera; 2]>() as u64),
			BufferUsage::none(),
		);
		let prev = data.prev;
		let curr = data.curr;
		let aspect = input.aspect;
		pass.build(move |mut pass| pass.write(buf, 0, &[GpuCamera::new(aspect, curr), GpuCamera::new(aspect, prev)]));
		Self { buf, prev, curr }
	}
}

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Camera {
	pub transform: Transform,
	pub camera: CameraComponent,
}

#[derive(Default)]
pub struct CameraSceneData {
	curr: Camera,
	prev: Camera,
}
impl Resource for CameraSceneData {}

fn find_primary_view(
	mut r: ResMut<CameraSceneData>, q: Query<(&Transform, &CameraComponent), With<PrimaryViewComponent>>,
) {
	let mut iter = q.iter();
	if let Some((t, c)) = iter.next() {
		r.prev = r.curr;
		r.curr = Camera {
			transform: *t,
			camera: *c,
		};
	} else {
		warn!("no primary view found, using default camera");
	}

	if iter.next().is_some() {
		warn!("multiple primary views found, using the first one");
	}
}
