use std::io;

use rad_core::{
	asset::{aref::ARef, Asset, AssetId, AssetView},
	Engine,
};
use rad_renderer::{
	assets::{
		material::Material,
		mesh::{GpuVertex, Mesh, MeshData},
	},
	components::{
		camera::{CameraComponent, PrimaryViewComponent},
		mesh::MeshComponent,
	},
	vek::{Vec2, Vec3},
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

	pub fn open_material(&mut self, id: AssetId) -> Result<(), io::Error> {
		let material: ARef<Material> = Engine::get().asset(id)?;

		// TODO: uhhhhhhhh
		let mut view = CursedAssetView { data: Vec::new() };
		Mesh::import(
			"test",
			MeshData {
				vertices: vec![
					GpuVertex {
						position: Vec3::new(-5.0, 0.0, 5.0),
						normal: Vec3::new(0.0, -1.0, 0.0),
						uv: Vec2::new(0.0, 0.0),
					},
					GpuVertex {
						position: Vec3::new(5.0, 0.0, 5.0),
						normal: Vec3::new(0.0, -1.0, 0.0),
						uv: Vec2::new(1.0, 0.0),
					},
					GpuVertex {
						position: Vec3::new(5.0, 0.0, -5.0),
						normal: Vec3::new(0.0, -1.0, 0.0),
						uv: Vec2::new(1.0, 1.0),
					},
					GpuVertex {
						position: Vec3::new(-5.0, 0.0, -5.0),
						normal: Vec3::new(0.0, -1.0, 0.0),
						uv: Vec2::new(0.0, 1.0),
					},
				],
				indices: vec![0, 1, 2, 0, 2, 3],
				material,
			},
			Box::new(&mut view),
		)
		.unwrap();
		let mesh = ARef::new(AssetId::new(), Mesh::load(Box::new(view)).unwrap());

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

struct CursedAssetView {
	data: Vec<u8>,
}

impl AssetView for CursedAssetView {
	fn name(&self) -> &str { "cursed" }

	fn clear(&mut self) -> Result<(), io::Error> {
		self.data.clear();
		Ok(())
	}

	fn new_section(&mut self) -> Result<Box<dyn io::Write + '_>, io::Error> { Ok(Box::new(&mut self.data)) }

	fn seek_begin(&mut self) -> Result<(), io::Error> { Ok(()) }

	fn read_section(&mut self) -> Result<Box<dyn io::Read + '_>, io::Error> { Ok(Box::new(&self.data[..])) }
}
