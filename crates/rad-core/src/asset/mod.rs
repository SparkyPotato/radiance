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
use petgraph::graph::{DiGraph, NodeIndex};
use rustc_hash::FxHashMap;
use tracing::trace_span;
pub use uuid::Uuid;

use crate::asset::aref::{AssetCache, AssetId, UntypedAssetId};

pub mod aref;

pub trait AssetRead: Read {}

pub trait AssetWrite: Write {}

/// An asset, representing *data* that can be loaded and saved.
pub trait Asset: Sized + 'static {
	const UUID: Uuid;
	/// The asset at the root of the cook chain.
	type Root: Asset = Self;

	/// Load the asset from the source. Note that this does *not* have to actually load everything
	/// into memory, it can just load metadata and stream.
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
		x => io::Error::other(format!("bincode error: {x:?}")),
	}
}

pub fn map_dec_err(e: DecodeError) -> io::Error {
	match e {
		DecodeError::Io { inner, .. } => inner,
		x => io::Error::other(format!("bincode error: {x:?}")),
	}
}

/// An asset derived from another.
pub trait CookedAsset: Asset {
	/// The asset this derives from.
	type Base: Asset<Root = Self::Root>;

	/// Cook the asset from its base.
	fn cook(base: &Self::Base) -> Self;
}

/// A runtime view of an asset. Most interactions with assets will be done through them.
pub trait AssetView: Sized + Send + Sync + 'static {
	/// The asset this view looks at.
	type Base: Asset;
	/// A global context shared by all views.
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
	base_layout: Layout,
	cook: fn(base: *const (), out: *mut ()),
}
type ErasedAssetLoad = fn(from: Box<dyn AssetRead>, out: *mut ()) -> Result<(), io::Error>;

pub struct AssetRegistry {
	cook_dep_graph: DiGraph<Uuid, ()>,
	asset_indices: FxHashMap<Uuid, NodeIndex>,
	sources: Vec<Box<dyn AssetSource>>,
	source_to_index: FxHashMap<TypeId, usize>,
	assets: FxHashMap<Uuid, ErasedAssetLoad>,
	view_caches: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
	kitchens: FxHashMap<Uuid, Kitchen>,
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetRegistry {
	pub fn new() -> Self {
		Self {
			cook_dep_graph: DiGraph::new(),
			asset_indices: FxHashMap::default(),
			sources: Vec::new(),
			source_to_index: FxHashMap::default(),
			assets: FxHashMap::default(),
			view_caches: FxHashMap::default(),
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
		let index = self.cook_dep_graph.add_node(T::UUID);
		self.asset_indices.insert(T::UUID, index);
	}

	pub fn register_cooked<T: CookedAsset>(&mut self) {
		self.register_asset::<T>();
		let base_index = self
			.asset_indices
			.get(&T::Base::UUID)
			.expect("base asset not registered");
		let cooked_index = self.asset_indices.get(&T::UUID).unwrap();
		self.cook_dep_graph.add_edge(*cooked_index, *base_index, ());

		self.kitchens.insert(
			T::UUID,
			Kitchen {
				base: T::Base::UUID,
				base_layout: Layout::new::<T::Base>(),
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
		self.view_caches
			.insert(TypeId::of::<T>(), Box::new(AssetCache::<T>::new()));
	}

	pub fn source<T: AssetSource>(&self) -> &T {
		match self.source_to_index.get(&TypeId::of::<T>()) {
			Some(&source) => unsafe { &*(self.sources[source].as_ref() as *const dyn AssetSource as *const T) },
			None => panic!("source `{}` not registered", std::any::type_name::<T>()),
		}
	}

	fn cache<T: AssetView>(&self) -> &AssetCache<T> {
		match self
			.view_caches
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

	pub fn cook_asset<T: CookedAsset>(&self, id: AssetId<T::Root>) -> Result<T, io::Error> {
		let base: T::Base = self.load_asset(id)?;
		let mut out = MaybeUninit::<T>::uninit();
		self.cook_dynamic(
			id.to_untyped(),
			T::UUID,
			&base as *const _ as _,
			out.as_mut_ptr() as *mut (),
		);
		Ok(unsafe { out.assume_init() })
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

		Err(io::Error::new(io::ErrorKind::NotFound, "asset not found"))
	}

	pub fn load_asset<T: Asset>(&self, id: AssetId<T::Root>) -> Result<T, io::Error> {
		let mut out = MaybeUninit::<T>::uninit();
		self.load_dynamic(id.to_untyped(), T::UUID, out.as_mut_ptr() as *mut ())?;
		Ok(unsafe { out.assume_init() })
	}
}
