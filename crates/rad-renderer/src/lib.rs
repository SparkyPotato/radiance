#![feature(let_chains)]

use rad_core::{asset::aref::ARef, Engine, EngineBuilder, Module};
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

use crate::{
	components::{
		camera::{CameraComponent, PrimaryViewComponent},
		light::LightComponent,
		mesh::MeshComponent,
	},
	scene::{map_transform, GpuTransform, Scene, SceneReader, SceneUpdater},
};

pub mod assets;
pub mod components;
pub mod debug;
pub mod mesh;
pub mod pt;
mod scene;
pub mod sky;
pub mod tonemap;
mod util;

pub struct RendererModule;

impl Module for RendererModule {
	fn init(engine: &mut EngineBuilder) {
		engine.asset::<assets::image::Image>();
		engine.asset::<assets::material::Material>();
		engine.asset::<assets::mesh::Mesh>();

		engine.component::<components::mesh::MeshComponent>();
		engine.component_dep_type::<Vec<ARef<assets::mesh::Mesh>>>();
		engine.component::<components::light::LightComponent>();
		engine.component::<components::camera::CameraComponent>();
		engine.component::<components::camera::PrimaryViewComponent>();

		let u = SceneUpdater::new(engine.get_global()).unwrap();
		engine.global(u);
	}
}

// TODO: better name
#[derive(Copy, Clone)]
pub struct PrimaryViewData {
	transform: GpuTransform,
	camera: CameraComponent,
	scene: SceneReader,
	id: WorldId,
}

pub struct WorldRenderer {
	scene: Scene,
	camera: CameraComponent,
	transform: GpuTransform,
	id: Option<WorldId>,
}
impl Resource for WorldRenderer {}

impl WorldRenderer {
	pub fn new() -> Result<Self> {
		Ok(Self {
			scene: Scene::new()?,
			camera: CameraComponent::default(),
			transform: GpuTransform::default(),
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
			transform: self.transform,
			camera: self.camera,
			scene,
			id: self
				.id
				.expect("`sync_scene` should be ticked befor calling `WorldRenderer::update`"),
		}
	}
}

fn sync_scene(
	mut r: ResMut<WorldRenderer>, qm: Query<(Entity, Ref<Transform>, Ref<MeshComponent>)>,
	ql: Query<(Entity, Ref<Transform>, Ref<LightComponent>)>, mut m: RemovedComponents<MeshComponent>,
	mut l: RemovedComponents<LightComponent>, id: WorldId,
) {
	r.id = Some(id);

	// TODO: very inefficient
	for (e, t, m) in qm.iter() {
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

	for (e, t, l) in ql.iter() {
		let added = l.is_added();
		// let t_changed = t.is_changed();
		// let l_changed = l.is_changed();

		if added {
			r.scene.add_light(e, &*t, &*l);
		} else {
			// r.scene.change_light_and_transform(e, &*t, &*l);
		}
	}

	for e in m.read() {
		r.scene.remove(e);
	}

	for e in l.read() {
		r.scene.remove_light(e);
	}
}

fn find_primary_view(mut r: ResMut<WorldRenderer>, q: Query<(Ref<Transform>, Ref<PrimaryViewComponent>)>) {
	let mut iter = q.iter();
	if let Some((t, c)) = iter.next() {
		r.camera = c.0;
		r.transform = map_transform(&t);
	} else {
		warn!("No primary view found, using default camera");
	}

	if let Some(_) = iter.next() {
		warn!("Multiple primary views found, using the first one");
	}
}
