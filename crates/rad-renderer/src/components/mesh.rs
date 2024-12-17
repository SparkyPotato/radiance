use rad_core::asset::aref::ARef;
use rad_world::RadComponent;

use crate::assets::mesh::Mesh;

#[derive(RadComponent)]
#[uuid("2a0f8a13-08ac-4bdc-ae62-467e40195445")]
pub struct MeshComponent {
	pub(crate) inner: Vec<ARef<Mesh>>,
}

impl MeshComponent {
	pub fn new(inner: &[ARef<Mesh>]) -> Self {
		Self {
			inner: inner.to_owned(),
		}
	}
}
