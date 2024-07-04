use std::time::Duration;

use bytemuck::{bytes_of, NoUninit};
use helpers::{cmd, load, pipeline, run, App, ShaderStage};
use radiance_graph::{
	ash::vk,
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
	},
	graph::{BufferDesc, BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, SwapchainImage},
	resource::{Image, ImageView, Resource},
};

struct Textures {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	image: Image,
	view: ImageView,
	sampler: vk::Sampler,
	sampler_id: SamplerId,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	image_id: ImageId,
	sampler_id: SamplerId,
	aspect_ratio: f32,
}

impl App for Textures {
	const NAME: &'static str = "textures";

	fn create(device: &Device) -> Self {
		let vertex = pipeline::compile(include_str!("vertex.wgsl"), ShaderStage::Vertex);
		let fragment = pipeline::compile(include_str!("fragment.wgsl"), ShaderStage::Fragment);
		let (pipeline, layout) = pipeline::simple(
			device,
			&vertex,
			&fragment,
			vk::Format::B8G8R8A8_SRGB,
			&[vk::PushConstantRange::builder()
				.size(std::mem::size_of::<PushConstants>() as _)
				.offset(0)
				.stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
				.build()],
		);

		let (image, view) = load::image(device, include_bytes!("red.webp"), load::ImageFormat::WebP);

		let sampler = unsafe {
			device
				.device()
				.create_sampler(
					&vk::SamplerCreateInfo::builder()
						.mag_filter(vk::Filter::LINEAR)
						.min_filter(vk::Filter::LINEAR),
					None,
				)
				.unwrap()
		};
		let sampler_id = device.descriptors().get_sampler(device, sampler);

		Self {
			pipeline,
			layout,
			image,
			view,
			sampler,
			sampler_id,
		}
	}

	fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_sampler(self.sampler, None);
			device.descriptors().return_sampler(self.sampler_id);
			self.view.destroy(device);
			self.image.destroy(device);
			device.device().destroy_pipeline_layout(self.layout, None);
			device.device().destroy_pipeline(self.pipeline, None);
		}
	}

	fn render<'frame>(&'frame mut self, frame: &mut Frame<'frame, '_, ()>, image: SwapchainImage, _: Duration) {
		let mut pass = frame.pass("triangle");

		let write = pass.output(
			image,
			ImageUsage {
				format: image.format,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				subresource: Default::default(),
			},
		);

		let index = pass.output(
			BufferDesc {
				size: std::mem::size_of::<u16>() as u64 * 6,
				upload: true,
			},
			BufferUsage {
				usages: &[BufferUsageType::IndexBuffer],
			},
		);

		pass.build(move |mut ctx| unsafe {
			let mut index = ctx.get(index);
			let b = bytes_of(&[0u16, 1, 3, 1, 2, 3]);
			index.data.as_mut()[..b.len()].copy_from_slice(b);

			let view = ctx.get(write);
			cmd::start_rendering_swapchain(ctx.device, ctx.buf, view, image.size);
			ctx.device
				.device()
				.cmd_bind_pipeline(ctx.buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

			ctx.device.device().cmd_push_constants(
				ctx.buf,
				self.layout,
				vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
				0,
				bytes_of(&PushConstants {
					image_id: self.view.id.unwrap(),
					sampler_id: self.sampler_id,
					aspect_ratio: image.size.width as f32 / image.size.height as f32,
				}),
			);
			ctx.device.device().cmd_bind_descriptor_sets(
				ctx.buf,
				vk::PipelineBindPoint::GRAPHICS,
				self.layout,
				0,
				&[ctx.device.descriptors().set()],
				&[],
			);
			ctx.device
				.device()
				.cmd_bind_index_buffer(ctx.buf, index.buffer, 0, vk::IndexType::UINT16);
			ctx.device.device().cmd_draw_indexed(ctx.buf, 6, 1, 0, 0, 0);

			ctx.device.device().cmd_end_rendering(ctx.buf);
		});
	}
}

fn main() { run::<Textures>() }
