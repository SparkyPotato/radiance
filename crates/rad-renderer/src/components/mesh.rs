use rad_core::asset::aref::AssetId;
use rad_world::RadComponent;

use crate::assets::{material::Material, mesh::Mesh};

#[derive(RadComponent)]
#[uuid("2a0f8a13-08ac-4bdc-ae62-467e40195445")]
pub struct MeshComponent {
	pub(crate) inner: Vec<(AssetId<Mesh>, AssetId<Material>)>,
}

impl MeshComponent {
	pub fn new(inner: &[(AssetId<Mesh>, AssetId<Material>)]) -> Self {
		Self {
			inner: inner.to_owned(),
		}
	}
}
