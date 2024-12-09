use std::io;

use rad_core::{asset::AssetId, Engine};
use rad_renderer::{
	components::camera::{CameraComponent, PrimaryViewComponent},
	WorldRenderer,
};
use rad_world::{tick::Tick, transform::Transform, Entity, World};

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

	pub fn edit_tick(&mut self) {
		self.edit
			.entity_mut(self.editor)
			.component_mut::<Transform>()
			.unwrap()
			.position
			.y += 0.01;
		self.edit_tick.tick(&mut self.edit);
	}

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
			.id();
	}
}
