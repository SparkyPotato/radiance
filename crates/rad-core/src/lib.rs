#![feature(associated_type_defaults)]
#![feature(coerce_unsized)]
#![feature(once_cell_try)]
#![feature(unsize)]

use std::{
	any::{Any, TypeId},
	sync::OnceLock,
};

use rustc_hash::FxHashMap;

use crate::asset::{aref::AssetId, Asset, AssetRegistry, AssetSource, AssetView, CookedAsset};

pub mod asset;

static ENGINE: OnceLock<Engine> = OnceLock::new();

pub struct Engine {
	assets: AssetRegistry,
	globals: GlobalRegistry,
}

impl Engine {
	pub fn builder() -> EngineBuilder { EngineBuilder::new() }

	pub fn get() -> &'static Engine {
		ENGINE
			.get()
			.expect("`Engine` not initialized, call `Engine::builder().build()` first")
	}

	pub fn global<T: Any + Send + Sync>(&self) -> &T { self.try_global().unwrap() }

	pub fn try_global<T: Any + Send + Sync>(&self) -> Option<&T> { self.globals.get() }

	pub fn asset_source<T: AssetSource>(&self) -> &T { self.assets.source::<T>() }

	pub fn load_asset<T: Asset>(&self, id: AssetId<T::Root>) -> Result<T, std::io::Error> { self.assets.load_asset(id) }

	pub fn cook_asset<T: CookedAsset>(&self, id: AssetId<T::Root>) -> Result<T, std::io::Error> {
		self.assets.cook_asset(id)
	}

	pub unsafe fn destroy() { std::ptr::drop_in_place(&ENGINE as *const _ as *mut OnceLock<Engine>); }
}

pub struct EngineBuilder {
	inner: Engine,
}

impl EngineBuilder {
	pub fn new() -> Self {
		Self {
			inner: Engine {
				assets: AssetRegistry::new(),
				globals: GlobalRegistry::new(),
			},
		}
	}

	pub fn global<T: Any + Send + Sync>(&mut self, value: T) { self.inner.globals.insert(value); }

	pub fn asset_source<T: AssetSource>(&mut self, source: T) { self.inner.assets.register_source(source); }

	pub fn asset<T: Asset>(&mut self) { self.inner.assets.register_asset::<T>(); }

	pub fn cooked_asset<T: CookedAsset>(&mut self) { self.inner.assets.register_cooked::<T>(); }

	pub fn asset_view<T: AssetView>(&mut self) { self.inner.assets.register_view::<T>(); }

	pub fn get_global<T: Any + Send + Sync>(&mut self) -> &mut T { self.inner.globals.get_mut().unwrap() }

	pub fn module<M: Module>(mut self) -> Self {
		M::init(&mut self);
		self
	}

	pub fn build(self) { ENGINE.get_or_init(|| self.inner); }
}

pub trait Module: 'static {
	fn init(engine: &mut EngineBuilder);
}

struct GlobalRegistry {
	values: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl GlobalRegistry {
	fn new() -> Self {
		Self {
			values: FxHashMap::default(),
		}
	}

	fn insert<T: Any + Send + Sync>(&mut self, value: T) { self.values.insert(TypeId::of::<T>(), Box::new(value)); }

	fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
		self.values.get(&TypeId::of::<T>()).map(|v| v.downcast_ref().unwrap())
	}

	fn get_mut<T: Any + Send + Sync>(&mut self) -> Option<&mut T> {
		self.values
			.get_mut(&TypeId::of::<T>())
			.map(|v| v.downcast_mut().unwrap())
	}
}

pub use uuid::uuid;
