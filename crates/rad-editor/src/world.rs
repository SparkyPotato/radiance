use std::io;

use rad_core::{Engine, asset::aref::AssetId};
use rad_renderer::{
	components::camera::{CameraComponent, PrimaryViewComponent},
	scene::register_all_gpu_scenes,
};
use rad_world::{
	World,
	bevy_ecs::{entity::Entity, world::EntityMut},
	serde::DoNotSerialize,
	tick::Tick,
};

pub struct WorldContext {
	edit: World,
	edit_tick: Tick,
	editor: Entity,
}

impl WorldContext {
	pub fn new() -> Self {
		let mut this = Self {
			edit: World::new(),
			edit_tick: Tick::new(),
			editor: Entity::from_raw(0),
		};
		this.setup_world();
		this
	}

	pub fn open(&mut self, id: AssetId<World>) -> Result<(), io::Error> {
		self.edit = Engine::get().load_asset(id)?;
		self.setup_world();

		Ok(())
	}

	pub fn editor_mut(&mut self) -> EntityMut<'_> { self.edit.entity_mut(self.editor).into() }

	pub fn edit_tick(&mut self) { self.edit_tick.tick(&mut self.edit); }

	pub fn world_mut(&mut self) -> &mut World { &mut self.edit }

	fn setup_world(&mut self) {
		self.edit_tick = Tick::new();
		self.editor = self
			.edit
			.spawn_empty()
			.insert((CameraComponent::default(), PrimaryViewComponent, DoNotSerialize))
			.id();
		// TODO: move somewhere else.
		register_all_gpu_scenes(&mut self.edit, &mut self.edit_tick);
	}
}
