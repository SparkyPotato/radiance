#![feature(let_chains)]

use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{graph::Frame, Result};
use rad_world::{
	system::{DetectChanges, Query, Ref, RemovedComponents, ResMut, Resource, WorldId},
	tick::Tick,
	transform::Transform,
	Entity,
	TickStage,
	World,
	WorldBuilderExt,
};
use tracing::warn;
pub use vek;
use vek::Mat4;

use crate::{
	components::{
		camera::{CameraComponent, PrimaryViewComponent},
		mesh::MeshComponent,
	},
	scene::{Scene, SceneReader, SceneUpdater},
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
		engine.asset::<assets::image::Image>();

		engine.component::<components::mesh::MeshComponent>();
		engine.component::<components::camera::CameraComponent>();
		engine.component::<components::camera::PrimaryViewComponent>();

		let u = SceneUpdater::new(engine.get_global()).unwrap();
		engine.global(u);
	}
}

#[derive(Copy, Clone)]
pub struct PrimaryViewData {
	view: Mat4<f32>,
	camera: CameraComponent,
	scene: SceneReader,
	id: WorldId,
}

pub struct WorldRenderer {
	scene: Scene,
	camera: CameraComponent,
	view: Mat4<f32>,
	id: Option<WorldId>,
}
impl Resource for WorldRenderer {}

impl WorldRenderer {
	pub fn new() -> Result<Self> {
		Ok(Self {
			scene: Scene::new()?,
			camera: CameraComponent::default(),
			view: Mat4::identity(),
			id: None,
		})
	}

	pub fn add_to_world(self, world: &mut World, tick: &mut Tick) {
		world.add_resource(self);
		tick.add_systems(TickStage::Render, (sync_scene, find_primary_view));
	}

	pub fn update<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, frame_index: u64) -> PrimaryViewData {
		let scene = Engine::get()
			.global::<SceneUpdater>()
			.update(frame, &mut self.scene, frame_index);

		PrimaryViewData {
			view: self.view,
			camera: self.camera,
			scene,
			id: self
				.id
				.expect("`sync_scene` should be ticked befor calling `WorldRenderer::update`"),
		}
	}
}

fn sync_scene(
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

fn find_primary_view(mut r: ResMut<WorldRenderer>, q: Query<(Ref<Transform>, Ref<PrimaryViewComponent>)>) {
	let mut iter = q.iter();
	if let Some((t, c)) = iter.next() {
		r.camera = c.0;
		r.view = t.into_matrix().inverted();
	} else {
		warn!("No primary view found, using default camera");
	}

	if let Some(_) = iter.next() {
		warn!("Multiple primary views found, using the first one");
	}
}
