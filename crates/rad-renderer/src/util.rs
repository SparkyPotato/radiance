use std::io::Write;

use bytemuck::{bytes_of, NoUninit};

pub struct SliceWriter<'a> {
	inner: &'a mut [u8],
}

impl<'a> SliceWriter<'a> {
	pub fn new(slice: &'a mut [u8]) -> Self { Self { inner: slice } }

	pub fn write<T: NoUninit>(&mut self, value: T) { self.inner.write(bytes_of(&value)).unwrap(); }

	pub fn write_slice<T: NoUninit>(&mut self, slice: &[T]) {
		let bytes = bytemuck::cast_slice(slice);
		self.inner.write(bytes).unwrap();
	}
}
