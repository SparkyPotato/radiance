#![feature(allocator_api)]

use rad_core::{EngineBuilder, Module, asset::aref::AssetId};
use rad_world::WorldBuilderExt;
pub use vek;

pub mod assets;
pub mod components;
pub mod debug;
pub mod mesh;
pub mod pt;
pub mod scene;
pub mod sky;
pub mod sort;
pub mod tonemap;
mod util;

pub struct RendererModule;

impl Module for RendererModule {
	fn init(engine: &mut EngineBuilder) {
		engine.asset::<assets::mesh::Mesh>();
		engine.asset::<assets::material::Material>();
		engine.cooked_asset::<assets::mesh::virtual_mesh::VirtualMesh>();
		engine.cooked_asset::<assets::image::ImageAsset>();

		engine.asset_view::<assets::mesh::RaytracingMeshView>();
		engine.asset_view::<assets::mesh::virtual_mesh::VirtualMeshView>();
		engine.asset_view::<assets::image::ImageAssetView>();
		engine.asset_view::<assets::material::MaterialView>();

		engine.component::<components::mesh::MeshComponent>();
		engine.component_dep_type::<Vec<AssetId<assets::mesh::Mesh>>>();
		engine.component::<components::light::LightComponent>();
		engine.component::<components::camera::CameraComponent>();
		engine.component::<components::camera::PrimaryViewComponent>();
	}
}
