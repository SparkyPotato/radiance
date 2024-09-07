use std::{
	fs::File,
	io::{BufReader, Write},
	path::Path,
};

use bincode::{config::standard, Decode, Encode};
use bytemuck::{bytes_of, NoUninit};
use zstd::{Decoder, Encoder};

use crate::AssetHeader;

pub struct Reader {
	decoder: Decoder<'static, BufReader<File>>,
}

impl Reader {
	pub fn from_file(file: File) -> Result<Self, std::io::Error> {
		Ok(Self {
			decoder: Decoder::new(file)?,
		})
	}

	pub fn deserialize<T: Decode>(mut self) -> Result<T, std::io::Error> {
		bincode::decode_from_std_read(&mut self.decoder, standard()).map_err(|x| std::io::Error::other(x))
	}
}

pub struct Writer {
	encoder: Encoder<'static, File>,
}

impl Writer {
	pub fn from_path(path: &Path, header: AssetHeader) -> Result<Self, std::io::Error> {
		let _ = std::fs::create_dir_all(path.parent().unwrap());
		let mut file = File::create(path)?;
		file.write_all(bytes_of(&header))?;
		Ok(Self {
			encoder: Encoder::new(file, 6)?,
		})
	}

	pub fn serialize<T: Encode>(mut self, data: T) -> Result<(), std::io::Error> {
		bincode::encode_into_std_write(data, &mut self.encoder, standard()).map_err(|x| std::io::Error::other(x))?;
		self.encoder.finish()?;
		Ok(())
	}
}

pub struct SliceWriter<'a> {
	bytes: &'a mut [u8],
}

impl<'a> SliceWriter<'a> {
	pub fn new(bytes: &'a mut [u8]) -> Self { Self { bytes } }
}

impl SliceWriter<'_> {
	#[track_caller]
	pub fn write_slice<T: NoUninit>(&mut self, slice: &[T]) -> Result<(), ()> {
		if self.bytes.len() < slice.len() * std::mem::size_of::<T>() {
			return Err(());
		}

		let slice: &[u8] = bytemuck::cast_slice(slice);
		let (write, rest) = std::mem::take(&mut self.bytes).split_at_mut(slice.len());
		write.copy_from_slice(slice);
		self.bytes = rest;

		Ok(())
	}

	#[track_caller]
	pub fn write<T: NoUninit>(&mut self, value: T) -> Result<(), ()> { self.write_slice(&[value]) }
}
