use bytemuck::{AnyBitPattern, NoUninit};

pub struct SliceReader<'a> {
	bytes: &'a [u8],
}

impl<'a> SliceReader<'a> {
	pub fn new(bytes: &'a [u8]) -> Self { Self { bytes } }

	pub fn read_slice<T: AnyBitPattern>(&mut self, len: usize) -> &[T] {
		let (read, rest) = self.bytes.split_at(len * std::mem::size_of::<T>());
		self.bytes = rest;
		bytemuck::cast_slice(read)
	}

	pub fn read<T: AnyBitPattern>(&mut self) -> T {
		let slice = self.read_slice(std::mem::size_of::<T>());
		*bytemuck::from_bytes(slice)
	}

	pub fn is_empty(&self) -> bool { self.bytes.is_empty() }
}

pub struct SliceWriter<'a> {
	bytes: &'a mut [u8],
}

impl<'a> SliceWriter<'a> {
	pub fn new(bytes: &'a mut [u8]) -> Self { Self { bytes } }
}

impl SliceWriter<'_> {
	pub fn write_slice<T: NoUninit>(&mut self, slice: &[T]) {
		let slice: &[u8] = bytemuck::cast_slice(slice);
		let (write, rest) = std::mem::take(&mut self.bytes).split_at_mut(slice.len());
		write.copy_from_slice(slice);
		self.bytes = rest;
	}

	pub fn write<T: NoUninit>(&mut self, value: T) {
		let slice = bytemuck::bytes_of(&value);
		self.write_slice(slice);
	}
}
