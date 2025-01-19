#![feature(allocator_api)]
#![feature(let_chains)]

use rad_core::{asset::aref::AssetId, EngineBuilder, Module};
use rad_world::WorldBuilderExt;
pub use vek;

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
		engine.component::<components::mesh::MeshComponent>();
		engine.component_dep_type::<Vec<AssetId<assets::mesh::Mesh>>>();
		engine.component::<components::light::LightComponent>();
		engine.component::<components::camera::CameraComponent>();
		engine.component::<components::camera::PrimaryViewComponent>();
	}
}
