use std::io::Write;

use bytemuck::{bytes_of, NoUninit};
use rad_graph::{
	device::Device,
	graph::{BufferUsage, ExternalBuffer, Frame, Res},
	resource::{Buffer, BufferDesc, BufferHandle, Resource},
	Result,
};

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

pub struct ResizableBuffer {
	name: String,
	inner: Buffer,
}

impl ResizableBuffer {
	pub fn new(device: &Device, name: &str, size: u64) -> Result<Self> {
		Ok(Self {
			inner: Buffer::create(
				device,
				BufferDesc {
					name,
					size,
					readback: false,
				},
			)?,
			name: name.to_string(),
		})
	}

	pub fn reserve(&mut self, frame: &mut Frame, pass_name: &str, size: u64) -> Result<Option<Res<BufferHandle>>> {
		let mut s = self.inner.size();
		if size > s {
			while size > s {
				s *= 2;
			}

			let old = std::mem::replace(
				&mut self.inner,
				Buffer::create(
					frame.device(),
					BufferDesc {
						name: &self.name,
						size: s,
						readback: false,
					},
				)?,
			);

			let mut pass = frame.pass(pass_name);
			let old_r = pass.resource(ExternalBuffer::new(&old), BufferUsage::transfer_read());
			let new_r = pass.resource(ExternalBuffer::new(&self.inner), BufferUsage::transfer_write());
			let size = old.size();
			pass.build(move |mut pass| pass.copy_buffer(old_r, new_r, 0, 0, size as _));
			frame.delete(old);
			Ok(Some(new_r))
		} else {
			Ok(None)
		}
	}
}
