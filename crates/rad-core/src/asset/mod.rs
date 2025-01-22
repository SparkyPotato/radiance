use std::{
	alloc::Layout,
	any::{Any, TypeId},
	io::{self, Read, Write},
	mem::MaybeUninit,
	sync::Arc,
};

use bincode::{
	error::{DecodeError, EncodeError},
	Decode,
	Encode,
};
use rustc_hash::FxHashMap;
use tracing::{trace_span, warn};
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

struct Kitchen {
	base: Uuid,
	layout: Layout,
	cook: fn(base: *const (), out: *mut ()),
}
type ErasedAssetLoad = fn(from: Box<dyn AssetRead>, out: *mut ()) -> Result<(), io::Error>;

pub struct AssetRegistry {
	cook_at_runtime: bool,
	sources: Vec<Box<dyn AssetSource>>,
	source_to_index: FxHashMap<TypeId, usize>,
	assets: FxHashMap<Uuid, ErasedAssetLoad>,
	views: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
	kitchens: FxHashMap<Uuid, Kitchen>,
}

impl AssetRegistry {
	pub fn new() -> Self {
		Self {
			cook_at_runtime: false,
			sources: Vec::new(),
			source_to_index: FxHashMap::default(),
			assets: FxHashMap::default(),
			views: FxHashMap::default(),
			kitchens: FxHashMap::default(),
		}
	}

	pub fn register_source<T: AssetSource>(&mut self, source: T) {
		let index = self.sources.len();
		self.source_to_index.insert(TypeId::of::<T>(), index);
		self.sources.push(Box::new(source));
	}

	pub fn register_asset<T: Asset>(&mut self) {
		self.assets.insert(T::UUID, |from, out| {
			let out = out as *mut T;
			unsafe {
				out.write(T::load(from)?);
			}
			Ok(())
		});
	}

	pub fn register_cooked<T: CookedAsset>(&mut self) {
		self.register_asset::<T>();
		self.kitchens.insert(
			T::UUID,
			Kitchen {
				base: T::Base::UUID,
				layout: Layout::new::<T>(),
				cook: |base, out| {
					let base = unsafe { &*(base as *const T::Base) };
					let out = out as *mut T;
					unsafe {
						out.write(T::cook(base));
					}
				},
			},
		);
	}

	pub fn register_view<T: AssetView>(&mut self) {
		self.views.insert(TypeId::of::<T>(), Box::new(AssetCache::<T>::new()));
	}

	pub fn cook_at_runtime(&mut self) { self.cook_at_runtime = true; }

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

	#[inline(always)]
	fn cook_dynamic(&self, id: UntypedAssetId, ty: Uuid, base: *const (), into: *mut ()) {
		let s = trace_span!("cook asset", id = %id, ty = %ty);
		let _e = s.enter();

		(self.kitchens.get(&ty).expect("asset not registered").cook)(base, into);
	}

	fn load_dynamic(&self, id: UntypedAssetId, ty: Uuid, into: *mut ()) -> Result<(), io::Error> {
		let s = trace_span!("load asset", id = %id, ty = %ty);
		let _e = s.enter();

		for src in self.sources.iter().rev() {
			match src.load(id, ty) {
				Ok(from) => {
					let load = self.assets.get(&ty).expect("asset not registered");
					load(from, into)?;
					return Ok(());
				},
				Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
				Err(e) => return Err(e),
			}
		}

		if let Some(kitchen) = self.kitchens.get(&ty) {
			if !self.cook_at_runtime {
				warn!(
					"runtime cooking is disabled, but asset (id={}, ty={}) not found, cooking at runtime instead",
					id, ty
				);
			}

			let base = unsafe { std::alloc::alloc(kitchen.layout) as _ };
			if let Err(e) = self.load_dynamic(id, kitchen.base, base) {
				unsafe { std::alloc::dealloc(base as _, kitchen.layout) };
				return Err(e);
			}
			self.cook_dynamic(id, ty, base, into);
			unsafe { std::alloc::dealloc(base as _, kitchen.layout) };
			return Ok(());
		}

		Err(io::Error::new(io::ErrorKind::NotFound, "asset not found"))
	}

	pub fn load_asset<T: Asset>(&self, id: AssetId<T::Root>) -> Result<T, io::Error> {
		let mut out = MaybeUninit::<T>::uninit();
		self.load_dynamic(id.to_untyped(), T::UUID, out.as_mut_ptr() as *mut ())?;
		Ok(unsafe { out.assume_init() })
	}
}
