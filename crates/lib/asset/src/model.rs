use bytemuck::bytes_of;
use uuid::Uuid;
use vek::Aabb;

use crate::util::SliceReader;

/// A model consisting of multiple meshes and a material assigned to each mesh.
#[derive(Clone)]
pub struct Model {
	pub meshes: Vec<Uuid>,
	pub aabb: Aabb<f32>,
}

impl Model {
	/// AABB, followed by array of UUIDs, compressed by zstd.
	pub(super) fn to_bytes(&self) -> Vec<u8> {
		let bytes: Vec<_> = bytes_of(&self.aabb.min)
			.iter()
			.chain(bytes_of(&self.aabb.max).iter())
			.chain(self.meshes.iter().flat_map(|x| x.as_bytes().iter()))
			.copied()
			.collect();
		zstd::encode_all(bytes.as_slice(), 8).unwrap()
	}

	pub(super) fn from_bytes(bytes: &[u8]) -> Result<Self, ()> {
		let bytes = zstd::decode_all(bytes).map_err(|_| ())?;
		let mut reader = SliceReader::new(&bytes);

		let min = reader.read().ok_or(())?;
		let max = reader.read().ok_or(())?;
		let bytes = reader.finish();
		let mut meshes = Vec::with_capacity(bytes.len() / 16);
		for chunk in bytes.chunks_exact(16) {
			meshes.push(Uuid::from_slice(chunk).unwrap());
		}

		Ok(Self {
			meshes,
			aabb: Aabb { min, max },
		})
	}
}
