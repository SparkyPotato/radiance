use std::{
	any::{Any, TypeId},
	io::{self, Read, Write},
};

use bincode::{
	error::{DecodeError, EncodeError},
	Decode,
	Encode,
};
use rustc_hash::FxHashMap;
pub use uuid::Uuid;

use crate::asset::aref::{AssetCache, AssetId, UntypedAssetId};

pub mod aref;

pub trait AssetRead: Read {
	fn section_count(&self) -> u32;

	fn seek_section(&self, id: u32) -> Result<(), io::Error>;
}

pub trait AssetWrite: Write {
	fn next_section(&mut self) -> u32;
}

pub trait Asset: Sized {
	const UUID: Uuid;
	type RealBase: Asset = Self;

	fn load(from: Box<dyn AssetRead>) -> Result<Self, io::Error>;

	fn save(&self, to: Box<dyn AssetWrite>) -> Result<(), io::Error>;
}

pub trait BincodeAsset: Encode + Decode + Sized {
	const UUID: Uuid;
	type RealBase: Asset = Self;
}
impl<T: BincodeAsset> Asset for T {
	type RealBase = T::RealBase;

	const UUID: Uuid = T::UUID;

	fn load(mut from: Box<dyn AssetRead>) -> Result<Self, io::Error> {
		let c = bincode::config::standard();
		bincode::decode_from_std_read(&mut from, c).map_err(map_dec_err)
	}

	fn save(&self, mut to: Box<dyn AssetWrite>) -> Result<(), io::Error> {
		let c = bincode::config::standard();
		bincode::encode_into_std_write(self, &mut to, c).map_err(map_enc_err)?;
		Ok(())
	}
}

pub fn map_enc_err(e: EncodeError) -> io::Error {
	match e {
		EncodeError::Io { inner, .. } => inner,
		x => io::Error::new(io::ErrorKind::Other, format!("bincode error: {x:?}")),
	}
}

pub fn map_dec_err(e: DecodeError) -> io::Error {
	match e {
		DecodeError::Io { inner, .. } => inner,
		x => io::Error::new(io::ErrorKind::Other, format!("bincode error: {x:?}")),
	}
}

pub trait CookedAsset: Asset {
	type Base: Asset;

	fn cook(base: &Self::Base) -> Self;
}

pub trait AssetView: Sized + Send + Sync + 'static {
	type Base: Asset;
	type Ctx: Default + Send + Sync + 'static;

	fn load(ctx: &'static Self::Ctx, base: Self::Base) -> Result<Self, io::Error>;
}

pub trait AssetSource: Send + Sync + 'static {
	fn load(&self, id: UntypedAssetId, ty: Uuid) -> Result<Box<dyn AssetRead>, io::Error>;
}

pub struct AssetRegistry {
	sources: Vec<Box<dyn AssetSource>>,
	views: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl AssetRegistry {
	pub fn new() -> Self {
		Self {
			sources: Vec::new(),
			views: FxHashMap::default(),
		}
	}

	pub fn register_source<T: AssetSource>(&mut self, source: T) { self.sources.push(Box::new(source)); }

	pub fn register_view<T: AssetView>(&mut self) {
		self.views.insert(TypeId::of::<T>(), Box::new(AssetCache::<T>::new()));
	}

	fn cache<T: AssetView>(&self) -> &AssetCache<T> {
		match self
			.views
			.get(&TypeId::of::<T>())
			.and_then(|cache| cache.downcast_ref::<AssetCache<T>>())
		{
			Some(cache) => cache,
			None => panic!("view `{}` not registered", std::any::type_name::<T>()),
		}
	}

	pub fn load_asset<T: Asset>(&self, id: AssetId<T::RealBase>) -> Result<T, io::Error> {
		for src in self.sources.iter().rev() {
			match src.load(id.to_untyped(), T::UUID) {
				Ok(from) => return T::load(from),
				Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
				Err(e) => return Err(e),
			}
		}

		Err(io::Error::new(io::ErrorKind::NotFound, "asset not found"))
	}
}
