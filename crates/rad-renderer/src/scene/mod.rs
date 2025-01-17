use std::{
	any::{Any, TypeId},
	marker::PhantomData,
};

use bytemuck::NoUninit;
use hashbrown::hash_map::Entry;
use rad_graph::{
	arena::Arena,
	graph::{ArenaMap, Frame},
};
use rad_world::{
	bevy_ecs::{
		system::{Res, Resource},
		world::unsafe_world_cell::UnsafeWorldCell,
	},
	tick::Tick,
	transform::Transform,
	World,
};
use vek::{Quaternion, Vec3};

pub mod camera;
pub mod light;
pub mod rt_scene;
pub mod virtual_scene;

pub trait GpuScene: Copy + 'static {
	type In;
	type Res: Resource;

	fn add_to_world(world: &mut World, tick: &mut Tick);

	fn update<'pass>(frame: &mut Frame<'pass, '_>, res: &'pass mut Self::Res, input: &Self::In) -> Self;
}

#[derive(Copy, Clone, Default, PartialEq, NoUninit)]
#[repr(C)]
pub struct GpuTransform {
	pub position: Vec3<f32>,
	pub rotation: Quaternion<f32>,
	pub scale: Vec3<f32>,
}

impl From<Transform> for GpuTransform {
	fn from(t: Transform) -> Self {
		Self {
			position: t.position,
			rotation: t.rotation,
			scale: t.scale,
		}
	}
}

#[derive(Default)]
struct SceneRunCondition<T: GpuScene> {
	run: bool,
	_phantom: PhantomData<fn() -> T>,
}
impl<T: GpuScene> Resource for SceneRunCondition<T> {}

fn should_scene_sync<T: GpuScene>(cond: Option<Res<SceneRunCondition<T>>>) -> bool {
	cond.map(|c| c.run).unwrap_or(false)
}

pub struct WorldRenderer<'pass, 'graph> {
	world: UnsafeWorldCell<'pass>,
	inputs: ArenaMap<'graph, TypeId, Box<dyn Any, &'graph Arena>>,
	scene_cache: ArenaMap<'graph, TypeId, Box<dyn Any, &'graph Arena>>,
}

impl<'pass, 'graph> WorldRenderer<'pass, 'graph> {
	pub fn new(world: &'pass mut World, arena: &'graph Arena) -> Self {
		Self {
			world: world.as_unsafe_world_cell(),
			inputs: ArenaMap::with_hasher_in(Default::default(), arena),
			scene_cache: ArenaMap::with_hasher_in(Default::default(), arena),
		}
	}

	pub fn set_input<T: GpuScene>(&mut self, input: T::In) {
		self.inputs
			.insert(TypeId::of::<T>(), Box::new_in(input, *self.inputs.allocator()));
	}

	pub fn get<T: GpuScene>(&mut self, frame: &mut Frame<'pass, '_>) -> T {
		let input = self.inputs.get(&TypeId::of::<T>()).unwrap();
		let arena = *self.scene_cache.allocator();
		match self.scene_cache.entry(TypeId::of::<T>()) {
			Entry::Occupied(e) => *e.get().downcast_ref::<T>().unwrap(),
			Entry::Vacant(e) => {
				let scene = T::update(
					frame,
					unsafe { self.world.get_resource_mut::<T::Res>().unwrap().into_inner() },
					input.downcast_ref::<T::In>().unwrap(),
				);
				e.insert(Box::new_in(scene, arena));
				scene
			},
		}
	}
}
