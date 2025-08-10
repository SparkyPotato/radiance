use std::marker::PhantomData;

use bytemuck::NoUninit;

use crate::{
	Result,
	device::{ComputePipeline, Device, RtPipeline, RtPipelineDesc, ShaderInfo},
	graph::{PassContext, Res},
	resource::BufferHandle,
};

pub struct ComputePass<T> {
	pipeline: ComputePipeline,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> ComputePass<T> {
	pub fn new(device: &Device, shader: ShaderInfo) -> Result<Self> {
		Ok(Self {
			pipeline: device.compute_pipeline(shader, false)?,
			_phantom: PhantomData,
		})
	}

	pub fn with_wave_32(device: &Device, shader: ShaderInfo) -> Result<Self> {
		Ok(Self {
			pipeline: device.compute_pipeline(shader, true)?,
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

	pub unsafe fn destroy(self) {
		unsafe {
			self.pipeline.destroy();
		}
	}
}

pub struct RtPass<T> {
	pipeline: RtPipeline,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> RtPass<T> {
	pub fn new(device: &Device, desc: RtPipelineDesc) -> Result<Self> {
		Ok(Self {
			pipeline: device.rt_pipeline(desc)?,
			_phantom: PhantomData,
		})
	}

	fn setup(&self, pass: &mut PassContext, push: &T) {
		pass.bind_rt(&self.pipeline);
		pass.push(0, push);
	}

	pub fn trace(&self, pass: &mut PassContext, push: &T, x: u32, y: u32, z: u32) {
		self.setup(pass, push);
		self.pipeline.trace_rays(pass.buf, x, y, z);
	}

	pub fn trace_indirect(&self, pass: &mut PassContext, push: &T, buf: Res<BufferHandle>, offset: usize) {
		self.setup(pass, push);
		let buf = pass.get(buf).addr;
		self.pipeline.trace_rays_indirect(pass.buf, buf + offset as u64);
	}

	pub unsafe fn destroy(self) {
		unsafe {
			self.pipeline.destroy();
		}
	}
}
