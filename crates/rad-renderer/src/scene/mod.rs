use rad_world::{tick::Tick, World};

pub trait GpuScene {
	type Reader;

	fn add_to_world(world: &mut World, tick: &mut Tick);
}
