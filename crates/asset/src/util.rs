use bytemuck::{AnyBitPattern, NoUninit};

pub struct SliceReader<'a> {
	bytes: &'a [u8],
}

impl<'a> SliceReader<'a> {
	pub fn new(bytes: &'a [u8]) -> Self { Self { bytes } }

	#[track_caller]
	pub fn read_slice<T: AnyBitPattern>(&mut self, len: usize) -> Option<&[T]> {
		if self.bytes.len() < len * std::mem::size_of::<T>() {
			return None;
		}

		let (read, rest) = self.bytes.split_at(len * std::mem::size_of::<T>());
		self.bytes = rest;
		Some(bytemuck::cast_slice(read))
	}

	#[track_caller]
	pub fn read<T: AnyBitPattern>(&mut self) -> Option<T> {
		let slice = self.read_slice(1)?;
		Some(slice[0])
	}

	pub fn finish(self) -> &'a [u8] { self.bytes }

	pub fn is_empty(&self) -> bool { self.bytes.is_empty() }
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
