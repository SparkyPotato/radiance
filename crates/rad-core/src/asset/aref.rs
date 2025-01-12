use std::{
	fmt::{Debug, Display},
	hash::Hash,
	io,
	marker::PhantomData,
	ops::Deref,
	sync::{Arc, OnceLock, RwLock},
};

use bevy_reflect::{Reflect, ReflectDeserialize, ReflectSerialize, TypePath};
use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{asset::AssetView, Engine};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Pod, Zeroable, Encode, Decode, Reflect)]
#[serde(transparent)]
#[repr(transparent)]
#[reflect(opaque, Serialize, Deserialize)]
pub struct UntypedAssetId(#[bincode(with_serde)] Uuid);

impl Debug for UntypedAssetId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl Display for UntypedAssetId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Serialize, Deserialize, Pod, Zeroable, Encode, Decode, Reflect)]
#[serde(transparent)]
#[repr(transparent)]
#[reflect(opaque, Serialize, Deserialize)]
#[reflect(type_path = false)]
#[reflect(where T: Sized)]
pub struct AssetId<T>(UntypedAssetId, PhantomData<fn() -> T>);

impl<T: 'static> TypePath for AssetId<T> {
	fn type_path() -> &'static str { std::any::type_name::<T>() }

	fn short_type_path() -> &'static str { std::any::type_name::<T>() }
}

impl<T> Copy for AssetId<T> {}
impl<T> Clone for AssetId<T> {
	fn clone(&self) -> Self { *self }
}

impl<T> PartialEq for AssetId<T> {
	fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl<T> Eq for AssetId<T> {}
impl<T> Hash for AssetId<T> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

impl<T> Debug for AssetId<T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl<T> Display for AssetId<T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl<T> AssetId<T> {
	pub fn new() -> Self { Self(UntypedAssetId(Uuid::new_v4()), PhantomData) }

	pub fn to_untyped(self) -> UntypedAssetId { self.0 }
}

struct ARefData<T: AssetView> {
	id: AssetId<T::Base>,
	data: OnceLock<T>,
}

// TODO: support weak references.
/// A unique reference to an asset view.
pub struct ARef<T: AssetView> {
	inner: Arc<ARefData<T>>,
}

impl<T: AssetView> ARef<T> {
	/// Creates an unloaded asset view reference. The returned reference might be loaded if the asset is already loaded.
	pub fn unloaded(id: AssetId<T::Base>) -> Self { Engine::get().assets().cache::<T>().unloaded(id) }

	/// Create a loaded asset view reference. This function will block until the asset view is loaded.
	pub fn loaded(id: AssetId<T::Base>) -> Result<LARef<T>, io::Error> {
		Engine::get().assets().cache::<T>().loaded(id)
	}

	/// Load the asset view. This function will block until the asset view is loaded.
	pub fn load(self) -> Result<LARef<T>, io::Error> {
		Engine::get().assets().cache::<T>().load(&self.inner)?;
		Ok(LARef { inner: self })
	}
}

/// A loaded asset view
pub struct LARef<T: AssetView> {
	inner: ARef<T>,
}

impl<T: AssetView> LARef<T> {
	pub fn into_inner(self) -> ARef<T> { self.inner }
}

impl<T: AssetView> Deref for LARef<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target { unsafe { self.inner.inner.data.get().unwrap_unchecked() } }
}

pub struct AssetCache<T: AssetView> {
	context: T::Ctx,
	loaded: RwLock<FxHashMap<AssetId<T::Base>, Arc<ARefData<T>>>>,
}

impl<T: AssetView> AssetCache<T> {
	pub fn new() -> Self {
		Self {
			context: T::Ctx::default(),
			loaded: RwLock::new(FxHashMap::default()),
		}
	}

	pub fn unloaded(&self, id: AssetId<T::Base>) -> ARef<T> {
		let read = self.loaded.read().unwrap();
		match read.get(&id) {
			Some(data) => ARef { inner: data.clone() },
			None => {
				drop(read);
				let mut write = self.loaded.write().unwrap();
				let inner = write
					.entry(id)
					.or_insert_with(|| {
						Arc::new(ARefData {
							id,
							data: OnceLock::new(),
						})
					})
					.clone();
				ARef { inner }
			},
		}
	}

	pub fn loaded(&'static self, id: AssetId<T::Base>) -> Result<LARef<T>, io::Error> {
		let inner = self.unloaded(id);
		self.load(&inner.inner)?;
		Ok(LARef { inner })
	}

	fn load<'a>(&'static self, inner: &'a ARefData<T>) -> Result<&'a T, io::Error> {
		inner.data.get_or_try_init(|| {
			let asset = Engine::get().assets().load_asset(inner.id)?;
			T::load(&self.context, asset)
		})
	}
}
