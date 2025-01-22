#![feature(trait_upcasting)]

use std::{
	any::TypeId,
	io,
	ops::{Deref, DerefMut},
};

pub use bevy_ecs;
use bevy_ecs::world::EntityWorldMut;
pub use bevy_reflect;
use bevy_reflect::{reflect_trait, FromType, GetTypeRegistration, Reflect, ReflectFromReflect, TypePath};
pub use rad_core::{asset::Uuid, uuid};
use rad_core::{
	asset::{map_dec_err, map_enc_err, Asset, AssetRead, AssetWrite},
	Engine,
	EngineBuilder,
	Module,
};
pub use rad_world_derive::RadComponent;
use rustc_hash::FxHashMap;

pub use crate::tick::TickStage;
use crate::{self as rad_world};

pub mod serde;
pub mod tick;
pub mod transform;

pub struct TypeRegistry {
	pub inner: bevy_reflect::TypeRegistry,
	uuid_map: FxHashMap<Uuid, TypeId>,
}

pub trait WorldBuilderExt {
	fn component<T: RadComponent + GetTypeRegistration>(&mut self);

	fn component_dep_type<T: Reflect + TypePath>(&mut self)
	where
		ReflectFromReflect: FromType<T>;
}

impl WorldBuilderExt for EngineBuilder {
	fn component<T: RadComponent + GetTypeRegistration>(&mut self) {
		let reg = self.get_global::<TypeRegistry>();
		reg.inner.register::<T>();
		reg.uuid_map.insert(T::uuid(), TypeId::of::<T>());
	}

	fn component_dep_type<T: Reflect + TypePath>(&mut self)
	where
		ReflectFromReflect: FromType<T>,
	{
		self.get_global::<TypeRegistry>()
			.inner
			.register_type_data::<T, ReflectFromReflect>();
	}
}

pub struct WorldModule;

impl Module for WorldModule {
	fn init(engine: &mut EngineBuilder) {
		engine.global(TypeRegistry {
			inner: bevy_reflect::TypeRegistry::new(),
			uuid_map: FxHashMap::default(),
		});

		engine.asset::<World>();

		engine.component::<transform::Transform>();
	}
}

#[reflect_trait]
pub trait RadComponent {
	fn uuid() -> Uuid
	where
		Self: Sized;

	fn uuid_dyn(&self) -> Uuid;
}

pub struct World {
	inner: bevy_ecs::world::World,
}

impl World {
	pub fn new() -> Self {
		Self {
			inner: bevy_ecs::world::World::new(),
		}
	}

	pub fn spawn_empty(&mut self) -> EntityWorldMut<'_> {
		let mut e = self.inner.spawn_empty();
		e.insert(transform::Transform::identity());
		e
	}
}

impl Deref for World {
	type Target = bevy_ecs::world::World;

	fn deref(&self) -> &Self::Target { &self.inner }
}
impl DerefMut for World {
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

impl Asset for World {
	const UUID: Uuid = uuid!("aac9bce6-582b-422b-b56c-2048cc0c4a2f");

	fn load(mut data: Box<dyn AssetRead>) -> Result<Self, io::Error> {
		let c = bincode::config::standard();

		let count: u32 = bincode::decode_from_std_read(&mut data, c).map_err(map_dec_err)?;
		let mut inner = bevy_ecs::world::World::new();
		for _ in 0..count {
			serde::deserialize_entity(&mut data, &mut inner)?;
		}

		Ok(Self { inner })
	}

	fn save(&self, mut to: &mut dyn AssetWrite) -> Result<(), io::Error> {
		let c = bincode::config::standard();
		let count = self.inner.entities().len();
		bincode::encode_into_std_write(count, &mut to, c).map_err(map_enc_err)?;

		for en in self.inner.iter_entities() {
			serde::serialize_entity(&mut to, &self.inner, en)?;
		}

		Ok(())
	}
}

fn ty_reg() -> &'static bevy_reflect::TypeRegistry { &Engine::get().global::<TypeRegistry>().inner }

fn uuid_to_ty(uuid: Uuid) -> Option<TypeId> { Engine::get().global::<TypeRegistry>().uuid_map.get(&uuid).copied() }
