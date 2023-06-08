use uuid::Uuid;

/// A model consisting of multiple meshes and a material assigned to each mesh.
#[derive(Clone)]
pub struct Model {
	pub meshes: Vec<Uuid>,
}

impl Model {
	/// Array of UUIDs compressed by zstd.
	pub(super) fn to_bytes(&self) -> Vec<u8> {
		let bytes: Vec<_> = self.meshes.iter().flat_map(|x| x.as_bytes().iter().copied()).collect();
		zstd::encode_all(bytes.as_slice(), 8).unwrap()
	}

	pub(super) fn from_bytes(bytes: &[u8]) -> Self {
		let bytes = zstd::decode_all(bytes).unwrap();
		let mut meshes = Vec::with_capacity(bytes.len() / 16);
		for chunk in bytes.chunks_exact(16) {
			meshes.push(Uuid::from_slice(chunk).unwrap());
		}
		Self { meshes }
	}
}
