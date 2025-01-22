use std::{
	any::{Any, TypeId},
	io::{self, Read, Write},
	sync::Arc,
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

pub trait AssetRead: Read {}

pub trait AssetWrite: Write {}

pub trait Asset: Sized + 'static {
	const UUID: Uuid;
	type Root: Asset = Self;

	fn load(from: Box<dyn AssetRead>) -> Result<Self, io::Error>;

	fn save(&self, to: &mut dyn AssetWrite) -> Result<(), io::Error>;
}

pub trait BincodeAsset: Encode + Decode + Sized + 'static {
	const UUID: Uuid;
	type Root: Asset = Self;
}
impl<T: BincodeAsset> Asset for T {
	type Root = T::Root;

	const UUID: Uuid = T::UUID;

	fn load(mut from: Box<dyn AssetRead>) -> Result<Self, io::Error> {
		let c = bincode::config::standard();
		bincode::decode_from_std_read(&mut from, c).map_err(map_dec_err)
	}

	fn save(&self, mut to: &mut dyn AssetWrite) -> Result<(), io::Error> {
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
	type Base: Asset<Root = Self::Root>;

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
impl<T: AssetSource> AssetSource for Arc<T> {
	fn load(&self, id: UntypedAssetId, ty: Uuid) -> Result<Box<dyn AssetRead>, io::Error> {
		T::load(self.as_ref(), id, ty)
	}
}

type ErasedKitchen = fn(base: *const (), out: *mut ());

pub struct AssetRegistry {
	sources: Vec<Box<dyn AssetSource>>,
	source_to_index: FxHashMap<TypeId, usize>,
	views: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
	kitchens: FxHashMap<TypeId, ErasedKitchen>,
}

impl AssetRegistry {
	pub fn new() -> Self {
		Self {
			sources: Vec::new(),
			source_to_index: FxHashMap::default(),
			views: FxHashMap::default(),
			kitchens: FxHashMap::default(),
		}
	}

	pub fn register_source<T: AssetSource>(&mut self, source: T) {
		let index = self.sources.len();
		self.source_to_index.insert(TypeId::of::<T>(), index);
		self.sources.push(Box::new(source));
	}

	pub fn register_view<T: AssetView>(&mut self) {
		self.views.insert(TypeId::of::<T>(), Box::new(AssetCache::<T>::new()));
	}

	pub fn register_cooked<T: CookedAsset>(&mut self) {
		self.kitchens.insert(TypeId::of::<T>(), |base, out| {
			let base = unsafe { &*(base as *const T::Base) };
			let out = out as *mut T;
			unsafe {
				out.write(T::cook(base));
			}
		});
	}

	pub fn source<T: AssetSource>(&self) -> &T {
		match self.source_to_index.get(&TypeId::of::<T>()) {
			Some(&source) => unsafe { &*(self.sources[source].as_ref() as *const dyn AssetSource as *const T) },
			None => panic!("source `{}` not registered", std::any::type_name::<T>()),
		}
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

	pub fn load_asset<T: Asset>(&self, id: AssetId<T::Root>) -> Result<T, io::Error> {
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
