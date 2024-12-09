#![feature(trait_upcasting)]

use std::{
	any::TypeId,
	io::{self, BufReader},
};

pub use bevy_ecs::{
	component::{Component, StorageType},
	entity::Entity,
	reflect::ReflectComponent,
};
use bevy_ecs::{system::Resource, world::Mut};
pub use bevy_reflect;
use bevy_reflect::{reflect_trait, GetTypeRegistration};
pub use rad_core::{asset::Uuid, uuid};
use rad_core::{
	asset::{Asset, AssetView},
	Engine,
	EngineBuilder,
	Module,
};
pub use rad_world_derive::RadComponent;
use rustc_hash::FxHashMap;

pub use crate::tick::TickStage;
use crate::{
	self as rad_world,
	serde::{map_dec_err, map_enc_err},
};

mod serde;
pub mod system;
pub mod tick;
pub mod transform;

pub struct TypeRegistry {
	inner: bevy_reflect::TypeRegistry,
	uuid_map: FxHashMap<Uuid, TypeId>,
}

pub trait WorldBuilderExt {
	fn component<T: RadComponent + GetTypeRegistration>(&mut self);
}

impl WorldBuilderExt for EngineBuilder {
	fn component<T: RadComponent + GetTypeRegistration>(&mut self) {
		let ty = TypeId::of::<T>();
		let uuid = T::uuid();
		self.get_global::<TypeRegistry>().inner.register::<T>();
		self.get_global::<TypeRegistry>().uuid_map.insert(uuid, ty);
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

pub struct EntityWrite<'a> {
	inner: bevy_ecs::world::EntityWorldMut<'a>,
}

impl EntityWrite<'_> {
	pub fn insert<T: RadComponent + Component>(&mut self, comp: T) -> &mut Self {
		self.inner.insert(comp);
		self
	}

	pub fn component_mut<T: RadComponent + Component>(&mut self) -> Option<&mut T> {
		self.inner.get_mut::<T>().map(|x| x.into_inner())
	}

	pub fn id(&self) -> Entity { self.inner.id() }
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

	pub fn spawn_empty(&mut self) -> EntityWrite<'_> {
		let mut inner = self.inner.spawn_empty();
		inner.insert(transform::Transform::identity());
		EntityWrite { inner }
	}

	pub fn add_resource<R: Resource>(&mut self, value: R) { self.inner.insert_resource(value); }

	pub fn get_resource<R: Resource>(&self) -> Option<&R> { self.inner.get_resource() }

	pub fn get_resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
		self.inner.get_resource_mut().map(|x: Mut<'_, R>| x.into_inner())
	}

	pub fn entity_mut(&mut self, e: Entity) -> EntityWrite<'_> {
		EntityWrite {
			inner: self.inner.entity_mut(e),
		}
	}
}

impl Asset for World {
	fn uuid() -> Uuid
	where
		Self: Sized,
	{
		uuid!("aac9bce6-582b-422b-b56c-2048cc0c4a2f")
	}

	fn load(mut data: Box<dyn AssetView>) -> Result<Self, io::Error>
	where
		Self: Sized,
	{
		let c = bincode::config::standard();
		data.seek_begin()?;
		let mut from = BufReader::new(data.read_section()?);

		let count: u32 = bincode::decode_from_reader(&mut from, c).map_err(map_dec_err)?;

		let mut inner = bevy_ecs::world::World::new();
		for _ in 0..count {
			let id = bincode::decode_from_reader(&mut from, c).map_err(map_dec_err)?;
			#[allow(deprecated)]
			let mut en = inner.get_or_spawn(bevy_ecs::entity::Entity::from_raw(id)).unwrap();
			let count: u32 = bincode::decode_from_reader(&mut from, c).map_err(map_dec_err)?;

			for _ in 0..count {
				serde::deserialize_component(&mut from, &mut en)?;
			}
		}

		Ok(Self { inner })
	}

	fn save(&self, into: &mut dyn AssetView) -> Result<(), io::Error> {
		into.clear()?;
		let mut into = into.new_section()?;

		let c = bincode::config::standard();
		let count = self.inner.entities().len();
		bincode::encode_into_std_write(count, &mut into, c).map_err(map_enc_err)?;

		for en in self.inner.iter_entities() {
			bincode::encode_into_std_write(en.id().index(), &mut into, c).map_err(map_enc_err)?;
			let count = en.archetype().component_count() as u32;
			bincode::encode_into_std_write(count, &mut into, c).map_err(map_enc_err)?;

			for comp in en.archetype().components() {
				let info = self.inner.components().get_info(comp).unwrap();
				serde::serialize_component(&mut into, en, info)?;
			}
		}

		Ok(())
	}
}

fn ty_reg() -> &'static bevy_reflect::TypeRegistry { &Engine::get().global::<TypeRegistry>().inner }

fn uuid_to_ty(uuid: Uuid) -> Option<TypeId> { Engine::get().global::<TypeRegistry>().uuid_map.get(&uuid).copied() }
