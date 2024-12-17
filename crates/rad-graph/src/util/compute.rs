use std::marker::PhantomData;

use bytemuck::NoUninit;

use crate::{
	device::{ComputePipeline, Device, ShaderInfo},
	graph::{PassContext, Res},
	resource::BufferHandle,
	Result,
};

pub struct ComputePass<T> {
	pipeline: ComputePipeline,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> ComputePass<T> {
	pub fn new(device: &Device, shader: ShaderInfo) -> Result<Self> {
		Ok(Self {
			pipeline: device.compute_pipeline(shader)?,
			_phantom: PhantomData,
		})
	}

	fn setup(&self, pass: &mut PassContext, push: &T) {
		pass.bind_compute(&self.pipeline);
		pass.push(0, push);
	}

	pub fn dispatch(&self, pass: &mut PassContext, push: &T, x: u32, y: u32, z: u32) {
		self.setup(pass, push);
		pass.dispatch(x, y, z);
	}

	pub fn dispatch_indirect(&self, pass: &mut PassContext, push: &T, buf: Res<BufferHandle>, offset: usize) {
		self.setup(pass, push);
		pass.dispatch_indirect(buf, offset);
	}

	pub unsafe fn destroy(self) { self.pipeline.destroy(); }
}
