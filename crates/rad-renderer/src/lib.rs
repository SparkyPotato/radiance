#![feature(let_chains)]

use rad_core::{EngineBuilder, Module};
use rad_world::{
	system::{DetectChanges, Query, Ref, RemovedComponents, ResMut, Resource},
	transform::Transform,
	Entity,
	WorldBuilderExt,
};
pub use vek;

use crate::{
	components::mesh::MeshComponent,
	debug::mesh::DebugMesh,
	mesh::VisBuffer,
	scene::{Scene, SceneUpdater},
};

pub mod assets;
pub mod components;
pub mod debug;
pub mod mesh;
mod scene;
mod util;

pub struct RendererModule;

impl Module for RendererModule {
	fn init(engine: &mut EngineBuilder) {
		engine.asset::<assets::mesh::Mesh>();
		engine.component::<components::mesh::MeshComponent>();

		let u = SceneUpdater::new(engine.get_global()).unwrap();
		engine.global(u);

		let v = VisBuffer::new(engine.get_global()).unwrap();
		engine.global(v);

		let d = DebugMesh::new(engine.get_global()).unwrap();
		engine.global(d);
	}
}

pub struct WorldRenderer {
	scene: Scene,
}
impl Resource for WorldRenderer {}

pub fn sync_scene(
	mut r: ResMut<WorldRenderer>, q: Query<(Entity, Ref<Transform>, Ref<MeshComponent>)>,
	mut m: RemovedComponents<MeshComponent>,
) {
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
