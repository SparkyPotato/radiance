//! This crate handles importing assets into our custom formats, as well as loading and storing them to disk.
//!
//! It provides raw access to assets, without any sort of caching or GPU resource management.

use std::{fmt::Debug, hash::BuildHasherDefault};

use bincode::{Decode, Encode};
use dashmap::DashMap;
use material::Material;
use rustc_hash::FxHasher;
use tracing::{event, Level};
pub use uuid::Uuid;

use crate::{
	mesh::Mesh,
	scene::Scene,
	util::{SliceReader, SliceWriter},
};

#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "import")]
pub mod import;
pub mod material;
pub mod mesh;
pub mod scene;
pub mod util;

pub type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

const CONTAINER_VERSION: u32 = 1;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// The type of an asset.
#[repr(u32)]
pub enum AssetType {
	Mesh = 1,
	Material = 2,
	Scene = 3,
}

impl TryFrom<u32> for AssetType {
	type Error = ();

	fn try_from(x: u32) -> Result<Self, ()> {
		Ok(match x {
			1 => Self::Mesh,
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
#[derive(Encode, Decode)]
pub enum Asset {
	Mesh(Mesh),
	Material(Material),
	Scene(Scene),
}

impl Asset {
	fn ty(&self) -> AssetType {
		match self {
			Self::Mesh(_) => AssetType::Mesh,
			Self::Material(_) => AssetType::Material,
			Self::Scene(_) => AssetType::Scene,
		}
	}

	fn to_bytes(&self) -> Vec<u8> {
		let mut v = Vec::new();
		let mut enc = zstd::Encoder::new(&mut v, 10).unwrap();
		let config = bincode::config::standard()
			.with_little_endian()
			.with_fixed_int_encoding();
		match self {
			Asset::Mesh(m) => bincode::encode_into_std_write(m, &mut enc, config),
			Asset::Material(m) => bincode::encode_into_std_write(m, &mut enc, config),
			Asset::Scene(s) => bincode::encode_into_std_write(s, &mut enc, config),
		}
		.unwrap();
		enc.finish().unwrap();
		v
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

pub enum HeaderParseError {
	InvalidMagic,
	InvalidVersion,
	InvalidType,
	LessThanHeaderSize,
}

impl Debug for HeaderParseError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::InvalidMagic => write!(f, "invalid magic"),
			Self::InvalidVersion => write!(f, "invalid version"),
			Self::InvalidType => write!(f, "invalid type"),
			Self::LessThanHeaderSize => write!(f, "less than header size"),
		}
	}
}

impl AssetHeader {
	/// Parse the header from a byte slice.
	///
	/// If `check_magic` is `true`, the first 6 bytes are checked to be "RADASS", and the total size of the slice must
	/// reflect this.
	///
	/// If `check_version` is `true`, the container version is checked to be equal to the current version, and the total
	/// size of the slice must reflect this.
	pub fn parse(bytes: &[u8], check_magic: bool, check_version: bool) -> Result<Self, HeaderParseError> {
		let mut reader = SliceReader::new(bytes);
		if check_magic {
			if reader.read_slice::<u8>(6).ok_or(HeaderParseError::LessThanHeaderSize)? != b"RADASS" {
				return Err(HeaderParseError::InvalidMagic);
			}
		}
		if check_version {
			if u32::from_le_bytes(
				reader
					.read_slice::<u8>(4)
					.ok_or(HeaderParseError::LessThanHeaderSize)?
					.try_into()
					.unwrap(),
			) != CONTAINER_VERSION
			{
				return Err(HeaderParseError::InvalidVersion);
			}
		}

		let uuid = Uuid::from_slice(reader.read_slice(16).ok_or(HeaderParseError::LessThanHeaderSize)?).unwrap();
		let ty = u32::from_le_bytes(
			reader
				.read_slice::<u8>(4)
				.ok_or(HeaderParseError::LessThanHeaderSize)?
				.try_into()
				.unwrap(),
		)
		.try_into()
		.map_err(|_| HeaderParseError::InvalidType)?;

		Ok(Self { uuid, ty })
	}

	pub fn to_bytes(self, with_magic: bool, with_version: bool) -> Vec<u8> {
		let len = with_magic.then_some(6).unwrap_or(0) + with_version.then_some(4).unwrap_or(0) + 16 + 4;
		let mut bytes = vec![0; len];
		let mut writer = SliceWriter::new(&mut bytes);
		if with_magic {
			writer.write_slice(b"RADASS").unwrap();
		}
		if with_version {
			writer.write(CONTAINER_VERSION).unwrap();
		}
		writer.write_slice(self.uuid.as_bytes()).unwrap();
		writer.write(u32::from(self.ty)).unwrap();
		bytes
	}
}

/// A source of an asset.
pub trait AssetSource {
	type Error: Debug;

	/// A human-readable name of the asset, if any.
	fn human_name(&self) -> Option<&str>;

	/// Load the header of the asset.
	fn load_header(&self) -> Result<Result<AssetHeader, HeaderParseError>, Self::Error>;

	/// Load the data of the asset.
	fn load_data(&self) -> Result<Vec<u8>, Self::Error>;
}

/// A sink of an asset.
///
/// Writing the header is not allowed, because the type or UUID of an asset should never change.
pub trait AssetSink {
	type Error;

	/// Write the data of the asset.
	fn write_data(&self, data: &[u8]) -> Result<(), Self::Error>;
}

/// Raw access to assets from sources and sinks.
pub struct AssetSystem<S> {
	assets: FxDashMap<Uuid, AssetMetadata<S>>,
}

impl<S: AssetSource> Default for AssetSystem<S> {
	fn default() -> Self {
		Self {
			assets: FxDashMap::default(),
		}
	}
}

pub enum AssetError<S: AssetSource> {
	Source(S::Error),
	InvalidHeader(HeaderParseError),
	InvalidAsset,
}

impl<S: AssetSource> Debug for AssetError<S> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Source(e) => write!(f, "source error: {:?}", e),
			Self::InvalidHeader(e) => write!(f, "invalid header: {:?}", e),
			Self::InvalidAsset => write!(f, "invalid asset"),
		}
	}
}

impl<S: AssetSource> AssetSystem<S> {
	pub fn new() -> Self { Self::default() }

	/// Add an asset from a source.
	pub fn add(&self, source: S) -> Result<AssetHeader, AssetError<S>> {
		let header = source
			.load_header()
			.map_err(|x| {
				event!(
					Level::ERROR,
					"failed to add invalid or inaccessible asset: `{}`",
					source.human_name().unwrap_or("unnamed asset")
				);

				AssetError::Source(x)
			})?
			.map_err(|x| {
				event!(
					Level::ERROR,
					"failed to add asset with invalid header: `{}`",
					source.human_name().unwrap_or("unnamed asset")
				);

				AssetError::InvalidHeader(x)
			})?;

		event!(
			Level::INFO,
			"type" = ?header.ty,
			uuid = ?header.uuid,
			"added asset: `{}`", source.human_name().unwrap_or("unnamed asset")
		);
		self.assets.insert(header.uuid, AssetMetadata { header, source });
		Ok(header)
	}

	/// Remove an asset.
	pub fn remove(&self, uuid: Uuid) -> (AssetHeader, S) {
		let (_, meta) = self.assets.remove(&uuid).unwrap();
		event!(
			Level::INFO,
			"type" = ?meta.header.ty,
			uuid = ?meta.header.uuid,
			"removed asset: `{}`", meta.source.human_name().unwrap_or("unnamed asset")
		);
		(meta.header, meta.source)
	}

	/// Load an asset.
	pub fn load(&self, uuid: Uuid) -> Result<Asset, AssetError<S>> {
		let meta = self
			.assets
			.get(&uuid)
			.unwrap_or_else(|| panic!("asset {:?} not found", uuid));
		let data = meta.source.load_data().map_err(|x| {
			event!(
				Level::ERROR,
				"failed to load invalid or inaccessible asset: `{}`",
				meta.source.human_name().unwrap_or("unnamed asset")
			);

			AssetError::Source(x)
		})?;
		let mut dec = zstd::Decoder::new(data.as_slice()).unwrap();
		let config = bincode::config::standard()
			.with_little_endian()
			.with_fixed_int_encoding();
		match meta.header.ty {
			AssetType::Mesh => Ok(Asset::Mesh(
				bincode::decode_from_std_read(&mut dec, config).map_err(|_| AssetError::InvalidAsset)?,
			)),
			AssetType::Material => Ok(Asset::Material(
				bincode::decode_from_std_read(&mut dec, config).map_err(|_| AssetError::InvalidAsset)?,
			)),
			AssetType::Scene => Ok(Asset::Scene(
				bincode::decode_from_std_read(&mut dec, config).map_err(|_| AssetError::InvalidAsset)?,
			)),
		}
	}

	pub fn metadata(&self, uuid: Uuid) -> Option<AssetHeader> { self.assets.get(&uuid).map(|x| x.header) }

	pub fn human_name(&self, uuid: Uuid) -> Option<String> {
		self.assets
			.get(&uuid)
			.and_then(|x| x.source.human_name().map(|x| x.to_string()))
	}

	pub fn assets_of_type(&self, ty: AssetType) -> impl Iterator<Item = Uuid> + '_ {
		self.assets.iter().filter_map(move |item| {
			if item.header.ty == ty {
				Some(item.header.uuid)
			} else {
				None
			}
		})
	}
}

impl<S: AssetSink> AssetSystem<S> {
	/// Make changes to an asset.
	///
	/// The asset must already exist, and `asset` must be of the same type.
	pub fn write(&self, uuid: Uuid, asset: Asset) -> Result<(), S::Error> {
		let meta = self.assets.get(&uuid).expect("asset does not exist");
		assert_eq!(meta.header.ty, asset.ty());
		meta.source.write_data(&asset.to_bytes())
	}
}

struct AssetMetadata<S> {
	header: AssetHeader,
	source: S,
}
