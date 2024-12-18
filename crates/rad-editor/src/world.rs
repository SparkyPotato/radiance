use std::io;

use rad_core::{
	asset::{aref::ARef, AssetId},
	Engine,
};
use rad_renderer::{
	assets::mesh::Mesh,
	components::{
		camera::{CameraComponent, PrimaryViewComponent},
		mesh::MeshComponent,
	},
	WorldRenderer,
};
use rad_world::{serde::DoNotSerialize, tick::Tick, Entity, EntityWrite, World};

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

	pub fn open(&mut self, id: AssetId) -> Result<(), io::Error> {
		self.edit = *Engine::get().asset_owned(id)?;
		self.setup_world();

		Ok(())
	}

	pub fn open_mesh(&mut self, id: AssetId) -> Result<(), io::Error> {
		let mesh: ARef<Mesh> = Engine::get().asset(id)?;
		self.edit = World::new();
		self.edit.spawn_empty().insert(MeshComponent::new(&[mesh]));
		self.setup_world();

		Ok(())
	}

	pub fn editor_mut(&mut self) -> EntityWrite<'_> { self.edit.entity_mut(self.editor) }

	pub fn edit_tick(&mut self) { self.edit_tick.tick(&mut self.edit); }

	pub fn renderer(&mut self) -> &mut WorldRenderer { self.edit.get_resource_mut().unwrap() }

	fn setup_world(&mut self) {
		self.edit_tick = Tick::new();
		WorldRenderer::new()
			.unwrap()
			.add_to_world(&mut self.edit, &mut self.edit_tick);
		self.editor = self
			.edit
			.spawn_empty()
			.insert(PrimaryViewComponent(CameraComponent::default()))
			.insert(DoNotSerialize)
			.id();
	}
}
