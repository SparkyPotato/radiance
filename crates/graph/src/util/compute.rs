use std::marker::PhantomData;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};

use crate::{
	device::{Device, Pipeline, ShaderInfo},
	graph::PassContext,
	Result,
};

pub struct ComputePass<T> {
	layout: vk::PipelineLayout,
	pipeline: Pipeline,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> ComputePass<T> {
	pub fn new(device: &Device, shader: ShaderInfo) -> Result<Self> {
		let layout = unsafe {
			device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::COMPUTE)
						.size(std::mem::size_of::<T>() as u32)]),
				None,
			)?
		};
		Ok(Self {
			layout,
			pipeline: device.compute_pipeline(layout, shader)?,
			_phantom: PhantomData,
		})
	}

	unsafe fn setup(&self, push: &T, pass: &PassContext) {
		let dev = pass.device.device();
		let buf = pass.buf;
		dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline.get());
		dev.cmd_bind_descriptor_sets(
			buf,
			vk::PipelineBindPoint::COMPUTE,
			self.layout,
			0,
			&[pass.device.descriptors().set()],
			&[],
		);
		dev.cmd_push_constants(buf, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_of(push));
	}

	pub fn dispatch(&self, push: &T, pass: &PassContext, x: u32, y: u32, z: u32) {
		unsafe {
			self.setup(push, pass);
			pass.device.device().cmd_dispatch(pass.buf, x, y, z);
		}
	}

	pub fn dispatch_indirect(&self, push: &T, pass: &PassContext, buf: vk::Buffer, offset: usize) {
		unsafe {
			self.setup(push, pass);
			pass.device.device().cmd_dispatch_indirect(pass.buf, buf, offset as _);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pipeline.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
