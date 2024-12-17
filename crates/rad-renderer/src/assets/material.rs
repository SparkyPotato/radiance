use std::io;

use bincode::{config::standard, Decode, Encode};
use rad_core::{
	asset::{aref::ARef, Asset, AssetView},
	uuid,
};
use rad_world::Uuid;
use vek::{Vec3, Vec4};

use crate::assets::image::Image;

#[derive(Encode, Decode)]
pub struct Material {
	#[bincode(with_serde)]
	pub base_color: Option<ARef<Image>>,
	#[bincode(with_serde)]
	pub base_color_factor: Vec4<f32>,
	#[bincode(with_serde)]
	pub metallic_roughness: Option<ARef<Image>>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	#[bincode(with_serde)]
	pub normal: Option<ARef<Image>>,
	#[bincode(with_serde)]
	pub emissive: Option<ARef<Image>>,
	#[bincode(with_serde)]
	pub emissive_factor: Vec3<f32>,
}

impl Asset for Material {
	fn uuid() -> Uuid
	where
		Self: Sized,
	{
		uuid!("15695530-bc12-4745-9410-21d24480e8f1")
	}

	fn load(mut data: Box<dyn AssetView>) -> Result<Self, io::Error>
	where
		Self: Sized,
	{
		data.seek_begin()?;
		bincode::decode_from_std_read(&mut data.read_section()?, standard())
			.map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
	}

	fn save(&self, into: &mut dyn AssetView) -> Result<(), io::Error> {
		into.clear()?;
		bincode::encode_into_std_write(self, &mut into.new_section()?, standard())
			.map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
		Ok(())
	}
}
