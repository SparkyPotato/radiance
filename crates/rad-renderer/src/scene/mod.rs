use std::{
	any::{Any, TypeId},
	marker::PhantomData,
};

use bytemuck::NoUninit;
use hashbrown::hash_map::Entry;
use rad_graph::{
	arena::Arena,
	graph::{ArenaMap, ArenaSet, Frame},
};
use rad_world::{
	bevy_ecs::{
		component::ComponentId,
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
#[repr(transparent)]
struct SceneRunCondition<T: GpuScene> {
	run: bool,
	_phantom: PhantomData<fn() -> T>,
}
impl<T: GpuScene> Resource for SceneRunCondition<T> {}

fn should_scene_sync<T: GpuScene>(cond: Option<Res<SceneRunCondition<T>>>) -> bool {
	cond.map(|c| c.run).unwrap_or(false)
}

// TODO: i think run conditions have one frame of latency.
pub struct WorldRenderer<'pass, 'graph> {
	world: UnsafeWorldCell<'pass>,
	inputs: ArenaMap<'graph, TypeId, Box<dyn Any, &'graph Arena>>,
	scene_cache: ArenaMap<'graph, TypeId, Box<dyn Any, &'graph Arena>>,
	unvisited: ArenaSet<'graph, ComponentId>,
}

pub fn register_gpu_scene<T: GpuScene>(world: &mut World, tick: &mut Tick) {
	world.insert_resource(SceneRunCondition::<T> {
		run: false,
		_phantom: PhantomData,
	});
	T::add_to_world(world, tick);
}

pub fn register_all_gpu_scenes(world: &mut World, tick: &mut Tick) {
	register_gpu_scene::<camera::CameraScene>(world, tick);
	register_gpu_scene::<light::LightScene>(world, tick);
	register_gpu_scene::<rt_scene::RtScene>(world, tick);
	register_gpu_scene::<virtual_scene::VirtualScene>(world, tick);
}

impl<'pass, 'graph> WorldRenderer<'pass, 'graph> {
	pub fn new(world: &'pass mut World, arena: &'graph Arena) -> Self {
		let mut unvisited = ArenaSet::with_hasher_in(Default::default(), arena);
		unvisited.insert(world.resource_id::<SceneRunCondition<camera::CameraScene>>().unwrap());
		unvisited.insert(world.resource_id::<SceneRunCondition<light::LightScene>>().unwrap());
		unvisited.insert(world.resource_id::<SceneRunCondition<rt_scene::RtScene>>().unwrap());
		unvisited.insert(
			world
				.resource_id::<SceneRunCondition<virtual_scene::VirtualScene>>()
				.unwrap(),
		);

		let mut this = Self {
			world: world.as_unsafe_world_cell(),
			inputs: ArenaMap::with_hasher_in(Default::default(), arena),
			scene_cache: ArenaMap::with_hasher_in(Default::default(), arena),
			unvisited,
		};
		this.set_input(());
		this
	}

	pub fn set_input<T: 'static>(&mut self, input: T) {
		self.inputs
			.insert(TypeId::of::<T>(), Box::new_in(input, *self.inputs.allocator()));
	}

	pub fn get<T: GpuScene>(&mut self, frame: &mut Frame<'pass, '_>) -> T {
		unsafe {
			self.world.get_resource_mut::<SceneRunCondition<T>>().unwrap().run = true;
			self.unvisited
				.remove(&self.world.components().resource_id::<SceneRunCondition<T>>().unwrap());
		}

		let input = self.inputs.get(&TypeId::of::<T::In>()).unwrap();
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

impl Drop for WorldRenderer<'_, '_> {
	fn drop(&mut self) {
		for &id in self.unvisited.iter() {
			unsafe {
				let ptr = self.world.get_resource_mut_by_id(id).unwrap();
				*ptr.with_type::<bool>() = false;
			}
		}
	}
}
