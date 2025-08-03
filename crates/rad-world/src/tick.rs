use bevy_ecs::schedule::{IntoSystemConfigs, IntoSystemSetConfigs, Schedule, SystemSet};

use crate::World;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, SystemSet)]
pub enum TickStage {
	PreUpdate,
	Update,
	PreRender,
	Render,
	PostRender,
}

pub struct Tick {
	inner: Schedule,
}

impl Default for Tick {
    fn default() -> Self {
        Self::new()
    }
}

impl Tick {
	pub fn new() -> Self {
		let mut inner = Schedule::default();
		inner.configure_sets((
			TickStage::PreUpdate.before(TickStage::Update),
			TickStage::Update.before(TickStage::PreRender),
			TickStage::PreRender.before(TickStage::Render),
			TickStage::Render.before(TickStage::PostRender),
		));
		Self { inner }
	}

	pub fn add_systems<M>(&mut self, stage: TickStage, systems: impl IntoSystemConfigs<M>) {
		self.inner.add_systems(systems.in_set(stage));
	}

	pub fn tick(&mut self, world: &mut World) {
		self.inner.run(&mut world.inner);
		world.inner.clear_trackers();
	}
}
