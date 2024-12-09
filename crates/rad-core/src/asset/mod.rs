use std::{
	any::{Any, TypeId},
	fmt::{Debug, Display},
	io::{self},
	sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
pub use uuid::Uuid;

use crate::asset::aref::{ARef, AWeak};

pub mod aref;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Pod, Zeroable)]
#[serde(transparent)]
#[repr(transparent)]
pub struct AssetId(Uuid);

impl Debug for AssetId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl Display for AssetId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl AssetId {
	pub fn new() -> Self { Self(Uuid::new_v4()) }
}

struct AssetDesc {
	load: fn(AssetId, Box<dyn AssetView>) -> Result<ARef<dyn Asset>, io::Error>,
	load_owned: fn(Box<dyn AssetView>) -> Result<Box<dyn Asset>, io::Error>,
}

enum CacheStatus {
	Loading,
	Loaded(AWeak<dyn Asset>),
}

pub struct AssetRegistry {
	assets: FxHashMap<Uuid, AssetDesc>,
	sources: FxHashMap<TypeId, Box<dyn AssetSource>>,
	cache: RwLock<FxHashMap<AssetId, CacheStatus>>,
}

fn load_asset<T: Asset>(id: AssetId, data: Box<dyn AssetView>) -> Result<ARef<dyn Asset>, io::Error> {
	let obj = T::load(data)?;
	Ok(ARef::new(id, obj))
}

fn load_asset_owned<T: Asset>(data: Box<dyn AssetView>) -> Result<Box<dyn Asset>, io::Error> {
	let obj = T::load(data)?;
	Ok(Box::new(obj) as Box<dyn Asset>)
}

impl AssetRegistry {
	pub fn new() -> Self {
		Self {
			assets: FxHashMap::default(),
			sources: FxHashMap::default(),
			cache: RwLock::new(FxHashMap::default()),
		}
	}

	pub fn register<T: Asset>(&mut self) {
		self.assets.insert(
			T::uuid(),
			AssetDesc {
				load: load_asset::<T>,
				load_owned: load_asset_owned::<T>,
			},
		);
	}

	pub fn source<T: AssetSource>(&mut self, source: T) { self.sources.insert(TypeId::of::<T>(), Box::new(source)); }

	pub fn get_source<T: AssetSource>(&self) -> Option<&T> {
		self.sources
			.get(&TypeId::of::<T>())
			.map(|s| (s.as_ref() as &dyn Any).downcast_ref().unwrap())
	}

	pub fn load_asset_dyn(&self, id: AssetId) -> Result<ARef<dyn Asset>, io::Error> {
		loop {
			let cache = self.cache.read();
			match cache.get(&id) {
				Some(CacheStatus::Loading) => {
					// TODO: this busy waits i think
					drop(cache);
					std::thread::yield_now();
				},
				Some(CacheStatus::Loaded(weak)) => {
					if let Some(asset) = weak.upgrade() {
						return Ok(asset);
					} else {
						break;
					}
				},
				None => break,
			}
		}

		self.cache.write().insert(id, CacheStatus::Loading);
		for source in self.sources.values() {
			match source.load(id) {
				Ok((uuid, data)) => {
					if let Some(desc) = self.assets.get(&uuid) {
						let asset = (desc.load)(id, data)?;
						self.cache
							.write()
							.insert(id, CacheStatus::Loaded(ARef::downgrade(&asset)));
						return Ok(asset);
					} else {
						return Err(io::Error::new(
							io::ErrorKind::NotFound,
							"unknown asset type (not registered?)",
						));
					}
				},
				Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
				Err(e) => return Err(e),
			}
		}

		Err(io::Error::new(io::ErrorKind::NotFound, "asset not found"))
	}

	pub fn load_asset_owned_dyn(&self, id: AssetId) -> Result<Box<dyn Asset>, io::Error> {
		for source in self.sources.values() {
			match source.load(id) {
				Ok((uuid, data)) => {
					if let Some(desc) = self.assets.get(&uuid) {
						return Ok((desc.load_owned)(data)?);
					} else {
						return Err(io::Error::new(
							io::ErrorKind::NotFound,
							"unknown asset type (not registered?)",
						));
					}
				},
				Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
				Err(e) => return Err(e),
			}
		}

		Err(io::Error::new(io::ErrorKind::NotFound, "asset not found"))
	}

	pub fn load_asset<T: Asset>(&self, id: AssetId) -> Result<ARef<T>, io::Error> {
		self.load_asset_dyn(id).and_then(|asset| {
			asset
				.downcast::<T>()
				.map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "asset type mismatch"))
		})
	}

	pub fn load_asset_owned<T: Asset>(&self, id: AssetId) -> Result<Box<T>, io::Error> {
		self.load_asset_owned_dyn(id).and_then(|asset| {
			Box::<dyn Any + Send + Sync>::downcast::<T>(asset as _)
				.map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "asset type mismatch"))
		})
	}
}

pub trait AssetView {
	fn clear(&mut self) -> Result<(), io::Error>;

	fn new_section(&mut self) -> Result<Box<dyn io::Write + '_>, io::Error>;

	fn seek_begin(&mut self) -> Result<(), io::Error>;

	fn read_section(&mut self) -> Result<Box<dyn io::Read + '_>, io::Error>;
}

pub trait Asset: Any + Send + Sync {
	fn uuid() -> Uuid
	where
		Self: Sized;

	fn load(data: Box<dyn AssetView>) -> Result<Self, io::Error>
	where
		Self: Sized;

	fn save(&self, into: &mut dyn AssetView) -> Result<(), io::Error>;
}

pub trait AssetSource: Any + Send + Sync {
	fn load(&self, id: AssetId) -> Result<(Uuid, Box<dyn AssetView>), io::Error>;
}

impl<T: AssetSource> AssetSource for Arc<T> {
	fn load(&self, id: AssetId) -> Result<(Uuid, Box<dyn AssetView>), io::Error> { self.as_ref().load(id) }
}
