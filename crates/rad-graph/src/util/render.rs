use std::marker::PhantomData;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};

use crate::{
	device::{Device, GraphicsPipelineDesc, Pipeline},
	graph::PassContext,
	Result,
};

pub struct RenderPass<T> {
	pipeline: Pipeline,
	flip: bool,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> RenderPass<T> {
	pub fn new(device: &Device, desc: GraphicsPipelineDesc, flip: bool) -> Result<Self> {
		Ok(Self {
			pipeline: device.graphics_pipeline(desc)?,
			flip,
			_phantom: PhantomData,
		})
	}

	pub fn run(
		&self, pass: &PassContext, push: &T, size: vk::Extent2D, attachments: &[vk::RenderingAttachmentInfo],
		draw: impl FnOnce(&Device, vk::CommandBuffer),
	) {
		let dev = pass.device.device();
		let buf = pass.buf;

		let area = vk::Rect2D::default().extent(size);

		unsafe {
			dev.cmd_begin_rendering(
				buf,
				&vk::RenderingInfo::default()
					.render_area(area)
					.layer_count(1)
					.color_attachments(attachments),
			);

			let width = size.width as f32;
			let height = size.height as f32;
			dev.cmd_set_viewport(
				buf,
				0,
				&[if self.flip {
					vk::Viewport {
						x: 0.0,
						y: height,
						width,
						height: -height,
						min_depth: 0.0,
						max_depth: 1.0,
					}
				} else {
					vk::Viewport {
						x: 0.0,
						y: 0.0,
						width,
						height,
						min_depth: 0.0,
						max_depth: 1.0,
					}
				}],
			);
			dev.cmd_set_scissor(buf, 0, &[area]);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline.get());
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				pass.device.layout(),
				0,
				&[pass.device.descriptors().set()],
				&[],
			);

			dev.cmd_push_constants(buf, pass.device.layout(), vk::ShaderStageFlags::ALL, 0, bytes_of(push));

			draw(pass.device, buf);

			dev.cmd_end_rendering(buf);
		}
	}

	pub fn run_empty(
		&self, pass: &PassContext, push: &T, size: vk::Extent2D, draw: impl FnOnce(&Device, vk::CommandBuffer),
	) {
		self.run(pass, push, size, &[], draw);
	}

	pub unsafe fn destroy(self) { self.pipeline.destroy(); }
}
