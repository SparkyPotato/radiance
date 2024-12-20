use std::{
	any::Any,
	hash::{Hash, Hasher},
	marker::Unsize,
	ops::{CoerceUnsized, Deref},
	sync::{Arc, Weak},
};

use bevy_reflect::{Reflect, ReflectDeserialize, ReflectSerialize, TypePath};
use serde::{Deserialize, Serialize};

use crate::{
	asset::{Asset, AssetId},
	Engine,
};

struct Data<T: ?Sized> {
	id: AssetId,
	inner: T,
}

#[derive(Reflect)]
#[reflect(opaque, Serialize, Deserialize)]
#[reflect(type_path = false)]
#[reflect(where T: Asset + Sized)]
pub struct ARef<T: ?Sized> {
	ptr: Arc<Data<T>>,
}

impl<T: Asset + ?Sized> Clone for ARef<T> {
	fn clone(&self) -> Self { Self { ptr: self.ptr.clone() } }
}

impl<T: Asset + ?Sized> Deref for ARef<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target { &self.ptr.inner }
}

impl<T: Asset + Unsize<U> + ?Sized, U: Asset + ?Sized> CoerceUnsized<ARef<U>> for ARef<T> {}

impl<T: Asset> ARef<T> {
	pub fn new(id: AssetId, obj: T) -> Self {
		Self {
			ptr: Arc::new(Data { id, inner: obj }),
		}
	}

	pub fn unloaded(id: AssetId) -> Self {
		Self {
			ptr: Arc::new(Data {
				id,
				inner: T::unloaded(),
			}),
		}
	}

	pub fn unknown() -> Self {
		Self {
			ptr: Arc::new(Data {
				id: AssetId::default(),
				inner: T::unloaded(),
			}),
		}
	}
}

impl<T: ?Sized + Asset> ARef<T> {
	pub fn asset_id(&self) -> AssetId { self.ptr.id }

	pub fn downgrade(&self) -> AWeak<T> {
		AWeak {
			ptr: Arc::downgrade(&self.ptr),
		}
	}
}

impl<T: Asset + ?Sized> ARef<T> {
	pub fn as_ref(&self) -> &T { &*self }
}

impl ARef<dyn Asset> {
	pub fn downcast<T: Asset>(self) -> Result<ARef<T>, ARef<dyn Asset>> {
		let ptr = Arc::into_raw(self.ptr);
		unsafe {
			if (*(&(*ptr).inner as *const dyn Any)).is::<T>() {
				Ok(ARef {
					ptr: Arc::from_raw(ptr as *const Data<T>),
				})
			} else {
				Err(ARef {
					ptr: Arc::from_raw(ptr),
				})
			}
		}
	}
}

impl<T: Asset> TypePath for ARef<T> {
	fn type_path() -> &'static str { "rad_core::asset::aref::ARef" }

	fn short_type_path() -> &'static str { "ARef" }
}
impl<T: Asset + ?Sized> Serialize for ARef<T> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		self.ptr.id.serialize(serializer)
	}
}
impl<'a, T: Asset> Deserialize<'a> for ARef<T> {
	fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
		let id = AssetId::deserialize(deserializer)?;
		Engine::get().asset(id).map_err(serde::de::Error::custom)
	}
}

#[derive(Reflect)]
#[reflect(opaque, Serialize, Deserialize)]
#[reflect(type_path = false)]
#[reflect(where T: Asset + Sized)]
pub struct AWeak<T: Asset + ?Sized> {
	ptr: Weak<Data<T>>,
}

impl<T: Asset + ?Sized> Clone for AWeak<T> {
	fn clone(&self) -> Self { Self { ptr: self.ptr.clone() } }
}
impl<T: Asset + ?Sized> Hash for AWeak<T> {
	fn hash<H: Hasher>(&self, state: &mut H) { self.ptr.as_ptr().hash(state); }
}
impl<T: Asset + ?Sized, U: Asset + ?Sized> PartialEq<AWeak<U>> for AWeak<T> {
	fn eq(&self, other: &AWeak<U>) -> bool { self.ptr.as_ptr().addr() == other.ptr.as_ptr().addr() }
}
impl<T: Asset + ?Sized> Eq for AWeak<T> {}

impl<T: Asset + Unsize<U> + ?Sized, U: Asset + ?Sized> CoerceUnsized<AWeak<U>> for AWeak<T> {}

impl<T: Asset + ?Sized> AWeak<T> {
	pub fn upgrade(&self) -> Option<ARef<T>> { self.ptr.upgrade().map(|ptr| ARef { ptr }) }
}

impl<T: Asset> TypePath for AWeak<T> {
	fn type_path() -> &'static str { "rad_core::asset::aref::AWeak" }

	fn short_type_path() -> &'static str { "AWeak" }
}
impl<T: Asset + ?Sized> Serialize for AWeak<T> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.serialize_unit() }
}
impl<'a, T: Asset> Deserialize<'a> for AWeak<T> {
	fn deserialize<D: serde::Deserializer<'a>>(_: D) -> Result<Self, D::Error> { Ok(AWeak { ptr: Weak::new() }) }
}
