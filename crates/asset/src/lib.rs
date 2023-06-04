//! This crate handles importing assets into our custom formats, as well as loading and storing them to disk.

use rustc_hash::FxHashMap;
use tracing::{event, Level};
use uuid::Uuid;

use crate::mesh::Mesh;

#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "import")]
pub mod import;
pub mod mesh;
mod util;

const CONTAINER_VERSION: u32 = 1;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// The type of an asset.
#[repr(u32)]
pub enum AssetType {
	Mesh = 0,
	Model = 1,
	Material = 2,
	Scene = 3,
}

impl TryFrom<u32> for AssetType {
	type Error = ();

	fn try_from(x: u32) -> Result<Self, ()> {
		Ok(match x {
			0 => Self::Mesh,
			1 => Self::Model,
			2 => Self::Material,
			3 => Self::Scene,
			_ => return Err(()),
		})
	}
}

impl From<AssetType> for u32 {
	fn from(v: AssetType) -> Self { unsafe { std::mem::transmute(v) } }
}

/// An asset.
pub enum Asset {
	Mesh(Mesh),
}

impl Asset {
	fn ty(&self) -> AssetType {
		match self {
			Self::Mesh(_) => AssetType::Mesh,
		}
	}

	fn to_bytes(&self) -> Vec<u8> {
		match self {
			Self::Mesh(mesh) => mesh.to_bytes(),
		}
	}
}

/// Asset header format:
/// 6 bytes: "RADASS"?
/// 4 bytes: container version?
/// 16 bytes: UUID
/// 4 bytes: asset type
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AssetHeader {
	/// UUID of the asset.
	pub uuid: Uuid,
	/// Type of the asset.
	pub ty: AssetType,
}

impl AssetHeader {
	/// Parse the header from a byte slice.
	///
	/// If `check_magic` is `true`, the first 6 bytes are checked to be "RADASS", and the total size of the slice must
	/// reflect this.
	///
	/// If `check_version` is `true`, the container version is checked to be equal to the current version, and the total
	/// size of the slice must reflect this.
	pub fn parse(bytes: &[u8], check_magic: bool, check_version: bool) -> Option<Self> {
		let mut offset = 0;
		if check_magic {
			if &bytes[0..6] != b"RADASS" {
				return None;
			}
			offset += 6;
		}
		if check_version {
			if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) != CONTAINER_VERSION {
				return None;
			}
			offset += 4;
		}

		let uuid = Uuid::from_slice(&bytes[offset..offset + 16]).unwrap();
		offset += 16;
		let ty = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
			.try_into()
			.ok()?;

		Some(Self { uuid, ty })
	}

	pub fn to_bytes(self, with_magic: bool, with_version: bool) -> Vec<u8> {
		let mut bytes = Vec::with_capacity(30);
		if with_magic {
			bytes.extend_from_slice(b"RADASS");
		}
		if with_version {
			bytes.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
		}
		bytes.extend_from_slice(self.uuid.as_bytes());
		bytes.extend_from_slice(&u32::from(self.ty).to_le_bytes());
		bytes
	}
}

/// A source of an asset.
pub trait AssetSource {
	type Error;

	/// A human-readable name of the asset, if any.
	fn human_name(&self) -> Option<&str>;

	/// Load the header of the asset. Return `None` if the header is invalid.
	fn load_header(&self) -> Result<AssetHeader, Self::Error>;

	/// Load the data of the asset. Return `None` if the data is invalid.
	fn load_data(&self) -> Result<Vec<u8>, Self::Error>;
}

/// A sink of an asset.
///
/// Writing the header is not allowed, because the type or UUID of an asset should never change.
pub trait AssetSink {
	type Error;

	/// Write the data of the asset.
	fn write_data(&mut self, data: &[u8]) -> Result<(), Self::Error>;
}

/// Raw access to assets from sources and sinks.
/// Does not handle any caching.
pub struct AssetSystem<S> {
	assets: FxHashMap<Uuid, AssetMetadata<S>>,
}

impl<S: AssetSource> Default for AssetSystem<S> {
	fn default() -> Self {
		Self {
			assets: FxHashMap::default(),
		}
	}
}

impl<S: AssetSource> AssetSystem<S> {
	pub fn new() -> Self { Self::default() }

	/// Add an asset from a source.
	pub fn add(&mut self, source: S) -> Result<AssetHeader, S::Error> {
		match source.load_header() {
			Ok(header) => {
				event!(
					Level::INFO,
					"type" = ?header.ty,
					uuid = ?header.uuid,
					"added asset: `{}`", source.human_name().unwrap_or("unnamed asset")
				);
				self.assets.insert(header.uuid, AssetMetadata { header, source });
				Ok(header)
			},
			Err(err) => {
				event!(
					Level::ERROR,
					"failed to add invalid or inaccessible asset: `{}`",
					source.human_name().unwrap_or("unnamed asset")
				);
				Err(err)
			},
		}
	}

	/// Remove an asset.
	pub fn remove(&mut self, uuid: Uuid) -> (AssetHeader, S) {
		let meta = self.assets.remove(&uuid).unwrap();
		event!(
			Level::INFO,
			"type" = ?meta.header.ty,
			uuid = ?meta.header.uuid,
			"removed asset: `{}`", meta.source.human_name().unwrap_or("unnamed asset")
		);
		(meta.header, meta.source)
	}

	/// Load an asset.
	pub fn load(&self, uuid: Uuid) -> Result<Asset, S::Error> {
		let meta = self.assets.get(&uuid).expect("asset does not exist");
		let data = meta.source.load_data()?;
		match meta.header.ty {
			AssetType::Mesh => Ok(Asset::Mesh(Mesh::from_bytes(&data))),
			_ => unimplemented!(),
		}
	}
}

impl<S: AssetSink> AssetSystem<S> {
	/// Make changes to an asset.
	///
	/// The asset must already exist, and `asset` must be of the same type.
	pub fn write(&mut self, uuid: Uuid, asset: Asset) -> Result<(), S::Error> {
		let meta = self.assets.get_mut(&uuid).expect("asset does not exist");
		assert_eq!(meta.header.ty, asset.ty());
		meta.source.write_data(&asset.to_bytes())
	}
}

struct AssetMetadata<S> {
	header: AssetHeader,
	source: S,
}
