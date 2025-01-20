use std::io;

use rad_core::{asset::aref::AssetId, Engine};
use rad_renderer::{
	assets::mesh::Mesh,
	components::{
		camera::{CameraComponent, PrimaryViewComponent},
		mesh::MeshComponent,
	},
	scene::register_all_gpu_scenes,
};
use rad_world::{
	bevy_ecs::{entity::Entity, world::EntityMut},
	serde::DoNotSerialize,
	tick::Tick,
	World,
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

	pub fn open_mesh(&mut self, id: AssetId<Mesh>) -> Result<(), io::Error> {
		self.edit = World::new();
		self.edit.spawn_empty().insert(MeshComponent::new(&[id]));
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
			.insert(PrimaryViewComponent(CameraComponent::default()))
			.insert(DoNotSerialize)
			.id();
		// TODO: move somewhere else.
		register_all_gpu_scenes(&mut self.edit, &mut self.edit_tick);
	}
}
