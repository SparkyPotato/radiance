#![feature(let_chains)]

use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{graph::Frame, Result};
use rad_world::{
	system::{DetectChanges, Query, Ref, RemovedComponents, ResMut, Resource, WorldId},
	transform::Transform,
	Entity,
	WorldBuilderExt,
};
pub use vek;

use crate::{
	components::mesh::MeshComponent,
	scene::{Scene, SceneReader, SceneUpdater},
};

pub mod assets;
pub mod components;
pub mod debug;
pub mod mesh;
pub mod scene;
mod util;

pub struct RendererModule;

impl Module for RendererModule {
	fn init(engine: &mut EngineBuilder) {
		engine.asset::<assets::mesh::Mesh>();
		engine.component::<components::mesh::MeshComponent>();

		let u = SceneUpdater::new(engine.get_global()).unwrap();
		engine.global(u);
	}
}

pub struct WorldRenderer {
	scene: Scene,
	id: Option<WorldId>,
}
impl Resource for WorldRenderer {}

impl WorldRenderer {
	pub fn new() -> Result<Self> {
		Ok(Self {
			scene: Scene::new()?,
			id: None,
		})
	}

	pub fn update<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, frame_index: u64) -> SceneReader {
		Engine::get().global::<SceneUpdater>().update(
			frame,
			&mut self.scene,
			self.id
				.expect("`sync_scene` should be ticked befor calling `WorldRenderer::update`"),
			frame_index,
		)
	}
}

pub fn sync_scene(
	mut r: ResMut<WorldRenderer>, q: Query<(Entity, Ref<Transform>, Ref<MeshComponent>)>,
	mut m: RemovedComponents<MeshComponent>, id: WorldId,
) {
	r.id = Some(id);

	// TODO: very inefficient
	for (e, t, m) in q.iter() {
		let added = m.is_added();
		let t_changed = t.is_changed();
		let m_changed = m.is_changed();

		if added {
			r.scene.add(e, &*t, &*m);
		} else if t_changed && m_changed {
			r.scene.change_mesh_and_transform(e, &*t, &*m);
		} else if t_changed {
			r.scene.change_transform(e, &*t);
		} else if m_changed {
			r.scene.change_mesh(e, &*m);
		}
	}

	for e in m.read() {
		r.scene.remove(e);
	}
}
