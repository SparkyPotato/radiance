use std::io;

use bevy_ecs::{
	component::{Component, ComponentInfo},
	entity::Entity,
	reflect::ReflectComponent,
	world::{EntityRef, EntityWorldMut, World},
};
use bevy_reflect::{
	DynamicArray,
	DynamicEnum,
	DynamicList,
	DynamicMap,
	DynamicSet,
	DynamicStruct,
	DynamicTuple,
	DynamicTupleStruct,
	DynamicVariant,
	Map,
	PartialReflect,
	ReflectDeserialize,
	ReflectFromReflect,
	ReflectRef,
	ReflectSerialize,
	Set,
	TypeInfo,
	TypeRegistration,
	VariantInfo,
};
use bincode::{
	de::Decoder,
	enc::Encoder,
	error::{DecodeError, EncodeError},
	serde::Compat,
	Decode,
	Encode,
};
use rad_core::asset::{map_dec_err, map_enc_err, Uuid};
use serde::{
	de::{DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess, Visitor},
	Deserializer,
};

use crate::{ty_reg, uuid_to_ty, ReflectRadComponent};

#[derive(Copy, Clone, Component)]
pub struct DoNotSerialize;

pub fn serialize_entity(mut into: &mut dyn io::Write, world: &World, en: EntityRef) -> Result<(), io::Error> {
	if en.contains::<DoNotSerialize>() {
		return Ok(());
	}

	let c = bincode::config::standard();
	bincode::encode_into_std_write(en.id().index(), &mut into, c).map_err(map_enc_err)?;
	let count = en.archetype().component_count() as u32;
	bincode::encode_into_std_write(count, &mut into, c).map_err(map_enc_err)?;

	for comp in en.archetype().components() {
		let info = world.components().get_info(comp).unwrap();
		serialize_component(&mut into, en, info)?;
	}

	Ok(())
}

fn serialize_component(mut into: &mut dyn io::Write, en: EntityRef, info: &ComponentInfo) -> Result<(), io::Error> {
	let c = bincode::config::standard();

	let Some(ty) = info.type_id() else { return Ok(()) };
	let reg = ty_reg().get(ty).ok_or_else(|| {
		io::Error::new(
			io::ErrorKind::InvalidData,
			format!("component (`{}`) not registered", info.name()),
		)
	})?;

	let Some(ref_rad) = reg.data::<ReflectRadComponent>() else {
		return Ok(());
	};
	let refl = reg
		.data::<ReflectComponent>()
		.ok_or_else(|| {
			io::Error::new(
				io::ErrorKind::InvalidData,
				format!("component (`{}`) not reflectable", info.name()),
			)
		})?
		.reflect(en)
		.unwrap();
	let uuid = (ref_rad.get_func)(refl).unwrap().uuid_dyn();

	bincode::encode_into_std_write(ComponentEncoder { uuid, comp: refl }, &mut into, c).map_err(map_enc_err)?;

	Ok(())
}

pub fn deserialize_entity(mut from: &mut dyn io::Read, world: &mut World) -> Result<(), io::Error> {
	let c = bincode::config::standard();
	let id = bincode::decode_from_std_read(&mut from, c).map_err(map_dec_err)?;
	#[allow(deprecated)]
	let mut en = world.get_or_spawn(bevy_ecs::entity::Entity::from_raw(id)).unwrap();
	let count: u32 = bincode::decode_from_std_read(&mut from, c).map_err(map_dec_err)?;

	for _ in 0..count {
		deserialize_component(&mut from, &mut en)?;
	}

	Ok(())
}

fn deserialize_component(mut from: &mut dyn io::Read, en: &mut EntityWorldMut) -> Result<(), io::Error> {
	let c = bincode::config::standard();
	let comp: CompenentDecoder = bincode::decode_from_std_read(&mut from, c).map_err(map_dec_err)?;
	comp.refl.insert(en, comp.obj.as_partial_reflect(), ty_reg());

	Ok(())
}

struct ComponentEncoder<'a> {
	uuid: Uuid,
	comp: &'a dyn PartialReflect,
}

impl Encode for ComponentEncoder<'_> {
	fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
		Encode::encode(&Compat(self.uuid), encoder)?;
		Encode::encode(&DynEncoder { val: self.comp }, encoder)?;

		Ok(())
	}
}

struct CompenentDecoder {
	refl: &'static ReflectComponent,
	obj: Box<dyn PartialReflect>,
}

impl Decode for CompenentDecoder {
	fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
		let uuid: Compat<Uuid> = Decode::decode(decoder)?;
		let id = uuid_to_ty(uuid.0).ok_or_else(|| DecodeError::Io {
			inner: io::Error::new(
				io::ErrorKind::InvalidData,
				format!("unknown component UUID (`{}`) not registered", uuid.0),
			),
			additional: 0,
		})?;
		let reg = ty_reg().get(id).unwrap();
		let refl = reg.data::<ReflectComponent>().ok_or_else(|| DecodeError::Io {
			inner: io::Error::new(
				io::ErrorKind::InvalidData,
				format!("component (`{}`) not reflectable", reg.type_info().type_path()),
			),
			additional: 0,
		})?;
		let obj = DynDecoder { reg }.decode(decoder)?;

		Ok(Self { refl, obj })
	}
}

struct DynEncoder<'a> {
	val: &'a dyn PartialReflect,
}

impl Encode for DynEncoder<'_> {
	fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
		if let Some::<&Entity>(e) = self.val.try_downcast_ref() {
			return Encode::encode(&e.index(), encoder);
		}

		match self.val.reflect_ref() {
			ReflectRef::Struct(x) => {
				for x in x.iter_fields() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::TupleStruct(x) => {
				for x in x.iter_fields() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::Tuple(x) => {
				for x in x.iter_fields() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::List(x) => {
				Encode::encode(&x.len(), encoder)?;
				for x in x.iter() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::Array(x) => {
				for x in x.iter() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::Map(x) => {
				Encode::encode(&x.len(), encoder)?;
				for (k, v) in x.iter() {
					Encode::encode(&DynEncoder { val: k }, encoder)?;
					Encode::encode(&DynEncoder { val: v }, encoder)?;
				}
			},
			ReflectRef::Set(x) => {
				Encode::encode(&x.len(), encoder)?;
				for x in x.iter() {
					Encode::encode(&DynEncoder { val: x }, encoder)?;
				}
			},
			ReflectRef::Enum(x) => {
				Encode::encode(&x.variant_index(), encoder)?;
				for x in x.iter_fields() {
					Encode::encode(&DynEncoder { val: x.value() }, encoder)?;
				}
			},
			ReflectRef::Opaque(_) => {
				let refl = self.val.try_as_reflect().unwrap();
				let se = ty_reg()
					.get_type_data::<ReflectSerialize>(refl.type_id())
					.ok_or_else(|| EncodeError::Io {
						inner: io::Error::new(io::ErrorKind::InvalidData, "opaque type not serializable"),
						index: 0,
					})?;

				Encode::encode(&Compat(&*se.get_serializable(refl)), encoder)?;
			},
		}

		Ok(())
	}
}

struct DynDecoder {
	reg: &'static TypeRegistration,
}

impl DynDecoder {
	fn decode(&self, decoder: &mut impl Decoder) -> Result<Box<dyn PartialReflect>, DecodeError> {
		if self.reg.type_info().is::<Entity>() {
			return Ok(Box::new(Entity::from_raw(Decode::decode(decoder)?)));
		}

		Ok(match self.reg.type_info() {
			TypeInfo::Struct(x) => {
				let mut s = DynamicStruct::default();
				for i in 0..x.field_len() {
					let reg = ty_reg().get(x.field_at(i).unwrap().ty().id()).unwrap();
					s.insert_boxed(x.field_names()[i], Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::TupleStruct(x) => {
				let mut s = DynamicTupleStruct::default();
				for i in 0..x.field_len() {
					let reg = ty_reg().get(x.field_at(i).unwrap().ty().id()).unwrap();
					s.insert_boxed(Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::Tuple(x) => {
				let mut s = DynamicTuple::default();
				for i in 0..x.field_len() {
					let reg = ty_reg().get(x.field_at(i).unwrap().ty().id()).unwrap();
					s.insert_boxed(Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::List(x) => {
				let mut s = DynamicList::default();
				let reg = ty_reg().get(x.item_ty().id()).unwrap();
				let len: usize = Decode::decode(decoder)?;
				for _ in 0..len {
					s.push_box(Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!(
								"component (`{}`) not unreflectable ({:?})",
								self.reg.type_info().type_path(),
								self.reg.type_id()
							),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::Array(x) => {
				let mut s = Vec::with_capacity(x.capacity());
				let reg = ty_reg().get(x.item_ty().id()).unwrap();
				for _ in 0..x.capacity() {
					s.push(Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&DynamicArray::new(s.into_boxed_slice()))
					.unwrap()
			},
			TypeInfo::Map(x) => {
				let mut s = DynamicMap::default();
				let key_reg = ty_reg().get(x.key_ty().id()).unwrap();
				let val_reg = ty_reg().get(x.value_ty().id()).unwrap();
				let len: usize = Decode::decode(decoder)?;
				for _ in 0..len {
					let key = Self { reg: key_reg }.decode(decoder)?;
					let val = Self { reg: val_reg }.decode(decoder)?;
					s.insert_boxed(key, val);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::Set(x) => {
				let mut s = DynamicSet::default();
				let reg = ty_reg().get(x.ty().id()).unwrap();
				let len: usize = Decode::decode(decoder)?;
				for _ in 0..len {
					s.insert_boxed(Self { reg }.decode(decoder)?);
				}
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::Enum(x) => {
				let i: usize = Decode::decode(decoder)?;
				let v = match x.variant_at(i).unwrap() {
					VariantInfo::Struct(x) => {
						let mut s = DynamicStruct::default();
						for i in 0..x.field_len() {
							let reg = ty_reg().get(x.field_at(i).unwrap().ty().id()).unwrap();
							s.insert_boxed(x.field_names()[i], Self { reg }.decode(decoder)?);
						}
						DynamicVariant::Struct(s)
					},
					VariantInfo::Tuple(x) => {
						let mut s = DynamicTuple::default();
						for i in 0..x.field_len() {
							let reg = ty_reg().get(x.field_at(i).unwrap().ty().id()).unwrap();
							s.insert_boxed(Self { reg }.decode(decoder)?);
						}
						DynamicVariant::Tuple(s)
					},
					VariantInfo::Unit(_) => DynamicVariant::Unit,
				};
				let s = DynamicEnum::new_with_index(i, x.variant_names()[i], v);
				self.reg
					.data::<ReflectFromReflect>()
					.ok_or_else(|| DecodeError::Io {
						inner: io::Error::new(
							io::ErrorKind::InvalidData,
							format!("component (`{}`) not unreflectable", self.reg.type_info().type_path()),
						),
						additional: 0,
					})?
					.from_reflect(&s)
					.unwrap()
			},
			TypeInfo::Opaque(_) => {
				let de = ty_reg()
					.get_type_data::<ReflectDeserialize>(self.reg.type_id())
					.expect("values must expose `ReflectDeserialize`");
				de.deserialize(SerdeDecoder { de: decoder })?.into_partial_reflect()
			},
		})
	}
}

use bincode::serde::DecodeError as SerdeDecodeError;

struct SerdeDecoder<'a, DE: Decoder> {
	de: &'a mut DE,
}

impl<'a, 'de, DE: Decoder> Deserializer<'de> for SerdeDecoder<'a, DE> {
	type Error = DecodeError;

	serde::serde_if_integer128! {
		fn deserialize_i128<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
		where
			V: serde::de::Visitor<'de>,
		{
			visitor.visit_i128(Decode::decode(&mut self.de)?)
		}
	}

	serde::serde_if_integer128! {
		fn deserialize_u128<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
		where
			V: serde::de::Visitor<'de>,
		{
			visitor.visit_u128(Decode::decode(&mut self.de)?)
		}
	}

	fn deserialize_any<V>(self, _: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		Err(SerdeDecodeError::AnyNotSupported.into())
	}

	fn deserialize_bool<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_bool(Decode::decode(&mut self.de)?)
	}

	fn deserialize_i8<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_i8(Decode::decode(&mut self.de)?)
	}

	fn deserialize_i16<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_i16(Decode::decode(&mut self.de)?)
	}

	fn deserialize_i32<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_i32(Decode::decode(&mut self.de)?)
	}

	fn deserialize_i64<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_i64(Decode::decode(&mut self.de)?)
	}

	fn deserialize_u8<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_u8(Decode::decode(&mut self.de)?)
	}

	fn deserialize_u16<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_u16(Decode::decode(&mut self.de)?)
	}

	fn deserialize_u32<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_u32(Decode::decode(&mut self.de)?)
	}

	fn deserialize_u64<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_u64(Decode::decode(&mut self.de)?)
	}

	fn deserialize_f32<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_f32(Decode::decode(&mut self.de)?)
	}

	fn deserialize_f64<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_f64(Decode::decode(&mut self.de)?)
	}

	fn deserialize_char<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_char(Decode::decode(&mut self.de)?)
	}

	fn deserialize_str<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_string(Decode::decode(&mut self.de)?)
	}

	fn deserialize_string<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_string(Decode::decode(&mut self.de)?)
	}

	fn deserialize_bytes<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_byte_buf(Decode::decode(&mut self.de)?)
	}

	fn deserialize_byte_buf<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_byte_buf(Decode::decode(&mut self.de)?)
	}

	fn deserialize_option<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		let is_some = u8::decode(&mut self.de)?;
		let variant = match is_some {
			0 => Ok(None),
			1 => Ok(Some(())),
			x => Err(DecodeError::UnexpectedVariant {
				found: x as u32,
				allowed: &bincode::error::AllowedEnumVariants::Range { max: 1, min: 0 },
				type_name: "Option<T>",
			}),
		}?;
		if variant.is_some() {
			visitor.visit_some(self)
		} else {
			visitor.visit_none()
		}
	}

	fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_unit()
	}

	fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_unit()
	}

	fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_newtype_struct(self)
	}

	fn deserialize_seq<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		let len = usize::decode(&mut self.de)?;
		self.deserialize_tuple(len, visitor)
	}

	fn deserialize_tuple<V>(mut self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		struct Access<'a, 'b, DE: Decoder> {
			deserializer: &'a mut SerdeDecoder<'b, DE>,
			len: usize,
		}

		impl<'de, 'a, 'b: 'a, DE: Decoder + 'b> SeqAccess<'de> for Access<'a, 'b, DE> {
			type Error = DecodeError;

			fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, DecodeError>
			where
				T: DeserializeSeed<'de>,
			{
				if self.len > 0 {
					self.len -= 1;
					let value = DeserializeSeed::deserialize(
						seed,
						SerdeDecoder {
							de: self.deserializer.de,
						},
					)?;
					Ok(Some(value))
				} else {
					Ok(None)
				}
			}

			fn size_hint(&self) -> Option<usize> { Some(self.len) }
		}

		visitor.visit_seq(Access {
			deserializer: &mut self,
			len,
		})
	}

	fn deserialize_tuple_struct<V>(self, _name: &'static str, len: usize, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		self.deserialize_tuple(len, visitor)
	}

	fn deserialize_map<V>(mut self, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		struct Access<'a, 'b, DE: Decoder> {
			deserializer: &'a mut SerdeDecoder<'b, DE>,
			len: usize,
		}

		impl<'de, 'a, 'b: 'a, DE: Decoder + 'b> MapAccess<'de> for Access<'a, 'b, DE> {
			type Error = DecodeError;

			fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, DecodeError>
			where
				K: DeserializeSeed<'de>,
			{
				if self.len > 0 {
					self.len -= 1;
					let key = DeserializeSeed::deserialize(
						seed,
						SerdeDecoder {
							de: self.deserializer.de,
						},
					)?;
					Ok(Some(key))
				} else {
					Ok(None)
				}
			}

			fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, DecodeError>
			where
				V: DeserializeSeed<'de>,
			{
				let value = DeserializeSeed::deserialize(
					seed,
					SerdeDecoder {
						de: self.deserializer.de,
					},
				)?;
				Ok(value)
			}

			fn size_hint(&self) -> Option<usize> { Some(self.len) }
		}

		let len = usize::decode(&mut self.de)?;

		visitor.visit_map(Access {
			deserializer: &mut self,
			len,
		})
	}

	fn deserialize_struct<V>(
		self, _name: &'static str, fields: &'static [&'static str], visitor: V,
	) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		self.deserialize_tuple(fields.len(), visitor)
	}

	fn deserialize_enum<V>(
		self, _name: &'static str, _variants: &'static [&'static str], visitor: V,
	) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		visitor.visit_enum(self)
	}

	fn deserialize_identifier<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		Err(SerdeDecodeError::IdentifierNotSupported.into())
	}

	fn deserialize_ignored_any<V>(self, _: V) -> Result<V::Value, Self::Error>
	where
		V: serde::de::Visitor<'de>,
	{
		Err(SerdeDecodeError::IgnoredAnyNotSupported.into())
	}

	fn is_human_readable(&self) -> bool { false }
}

impl<'de, 'a, DE: Decoder> EnumAccess<'de> for SerdeDecoder<'a, DE> {
	type Error = DecodeError;
	type Variant = Self;

	fn variant_seed<V>(mut self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>
	where
		V: DeserializeSeed<'de>,
	{
		let idx = u32::decode(&mut self.de)?;
		let val = seed.deserialize(idx.into_deserializer())?;
		Ok((val, self))
	}
}

impl<'de, 'a, DE: Decoder> VariantAccess<'de> for SerdeDecoder<'a, DE> {
	type Error = DecodeError;

	fn unit_variant(self) -> Result<(), Self::Error> { Ok(()) }

	fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
	where
		T: DeserializeSeed<'de>,
	{
		DeserializeSeed::deserialize(seed, self)
	}

	fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
	where
		V: Visitor<'de>,
	{
		Deserializer::deserialize_tuple(self, len, visitor)
	}

	fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value, Self::Error>
	where
		V: Visitor<'de>,
	{
		Deserializer::deserialize_tuple(self, fields.len(), visitor)
	}
}
