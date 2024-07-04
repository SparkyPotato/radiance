use std::time::Duration;

use helpers::{cmd, pipeline, run, App, ShaderStage};
use radiance_graph::{
	ash::vk,
	device::Device,
	graph::{Frame, ImageUsage, ImageUsageType, SwapchainImage},
};

struct Triangle(vk::Pipeline);

impl App for Triangle {
	const NAME: &'static str = "triangle";

	fn create(device: &Device) -> Self {
		let vertex = pipeline::compile(include_str!("vertex.wgsl"), ShaderStage::Vertex);
		let fragment = pipeline::compile(include_str!("fragment.wgsl"), ShaderStage::Fragment);
		let (pipeline, layout) = pipeline::simple(device, &vertex, &fragment, vk::Format::B8G8R8A8_SRGB, &[]);
		unsafe {
			device.device().destroy_pipeline_layout(layout, None);
		}

		Self(pipeline)
	}

	fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_pipeline(self.0, None);
		}
	}

	fn render<'frame>(&'frame mut self, frame: &mut Frame<'frame, '_, ()>, image: SwapchainImage, _: Duration) {
		let mut pass = frame.pass("triangle");

		let write = pass.output(
			image,
			ImageUsage {
				format: image.format,
				view_type: vk::ImageViewType::TYPE_2D,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				subresource: Default::default(),
			},
		);

		pass.build(move |mut ctx| unsafe {
			let view = ctx.get(write);
			cmd::start_rendering_swapchain(ctx.device, ctx.buf, view, image.size);
			ctx.device
				.device()
				.cmd_bind_pipeline(ctx.buf, vk::PipelineBindPoint::GRAPHICS, self.0);
			ctx.device.device().cmd_draw(ctx.buf, 3, 1, 0, 0);
			ctx.device.device().cmd_end_rendering(ctx.buf);
		});
	}
}

fn main() { run::<Triangle>() }
