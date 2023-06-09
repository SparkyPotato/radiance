//! This crate handles importing assets into our custom formats, as well as loading and storing them to disk.
//!
//! It provides raw access to assets, without any sort of caching or GPU resource management.

use std::fmt::Debug;

use rustc_hash::FxHashMap;
use tracing::{event, Level};
pub use uuid::Uuid;

use crate::{
	mesh::Mesh,
	model::Model,
	scene::Scene,
	util::{SliceReader, SliceWriter},
};

#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "import")]
pub mod import;
pub mod mesh;
pub mod model;
pub mod scene;
pub mod util;

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
	fn from(v: AssetType) -> Self { v as u32 }
}

/// An asset.
pub enum Asset {
	Mesh(Mesh),
	Model(Model),
	Scene(Scene),
}

impl Asset {
	fn ty(&self) -> AssetType {
		match self {
			Self::Mesh(_) => AssetType::Mesh,
			Self::Model(_) => AssetType::Model,
			Self::Scene(_) => AssetType::Scene,
		}
	}

	fn to_bytes(&self) -> Vec<u8> {
		match self {
			Self::Mesh(mesh) => mesh.to_bytes(),
			Self::Model(model) => model.to_bytes(),
			Self::Scene(scene) => scene.to_bytes(),
		}
	}
}

/// Asset header format:
/// 6 bytes: "RADASS"?
/// 4 bytes: container version?
/// 16 bytes: UUID
/// 4 bytes: asset type
#[derive(Copy, Clone, Debug)]
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
		let mut reader = SliceReader::new(bytes);
		if check_magic {
			if reader.read_slice::<u8>(6) != b"RADASS" {
				return None;
			}
		}
		if check_version {
			if u32::from_le_bytes(reader.read_slice::<u8>(4).try_into().unwrap()) != CONTAINER_VERSION {
				return None;
			}
		}

		let uuid = Uuid::from_slice(reader.read_slice(16)).unwrap();
		let ty = u32::from_le_bytes(reader.read_slice::<u8>(4).try_into().unwrap())
			.try_into()
			.ok()?;

		Some(Self { uuid, ty })
	}

	pub fn to_bytes(self, with_magic: bool, with_version: bool) -> Vec<u8> {
		let len = with_magic.then_some(6).unwrap_or(0) + with_version.then_some(4).unwrap_or(0) + 16 + 4;
		let mut bytes = vec![0; len];
		let mut writer = SliceWriter::new(&mut bytes);
		if with_magic {
			writer.write_slice(b"RADASS");
		}
		if with_version {
			writer.write(CONTAINER_VERSION)
		}
		writer.write_slice(self.uuid.as_bytes());
		writer.write(u32::from(self.ty));
		bytes
	}
}

/// A source of an asset.
pub trait AssetSource {
	type Error: Debug;

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
		let meta = self
			.assets
			.get(&uuid)
			.unwrap_or_else(|| panic!("asset {:?} not found", uuid));
		let data = meta.source.load_data()?;
		match meta.header.ty {
			AssetType::Mesh => Ok(Asset::Mesh(Mesh::from_bytes(&data))),
			AssetType::Model => Ok(Asset::Model(Model::from_bytes(&data))),
			AssetType::Scene => Ok(Asset::Scene(Scene::from_bytes(&data))),
			_ => unimplemented!(),
		}
	}

	pub fn assets_of_type(&self, ty: AssetType) -> impl Iterator<Item = Uuid> + '_ {
		self.assets.iter().filter_map(
			move |(uuid, meta)| {
				if meta.header.ty == ty {
					Some(*uuid)
				} else {
					None
				}
			},
		)
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
