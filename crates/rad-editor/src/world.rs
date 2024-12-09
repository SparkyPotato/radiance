use std::io;

use rad_core::{asset::AssetId, Engine};
use rad_renderer::{sync_scene, WorldRenderer};
use rad_world::{tick::Tick, TickStage, World};

pub struct WorldContext {
	edit: World,
	edit_tick: Tick,
}

impl WorldContext {
	pub fn new() -> Self {
		let mut this = Self {
			edit: World::new(),
			edit_tick: Tick::new(),
		};
		this.setup_world();
		this
	}

	pub fn open(&mut self, id: AssetId) -> Result<(), io::Error> {
		self.edit = *Engine::get().asset_owned(id)?;
		self.setup_world();

		Ok(())
	}

	pub fn edit_tick(&mut self) { self.edit_tick.tick(&mut self.edit); }

	pub fn renderer(&mut self) -> &mut WorldRenderer { self.edit.get_resource_mut().unwrap() }

	fn setup_world(&mut self) {
		self.edit.add_resource(WorldRenderer::new().unwrap());

		self.edit_tick = Tick::new().add_systems(TickStage::Render, sync_scene);
	}
}
