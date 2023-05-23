use std::time::Duration;

use bytemuck::{bytes_of, NoUninit};
use helpers::{cmd, load, pipeline, run, App, RenderInput, ShaderStage};
use radiance_graph::{
	ash::vk::{
		Filter,
		Format,
		ImageAspectFlags,
		ImageViewType,
		IndexType,
		Pipeline,
		PipelineBindPoint,
		PipelineLayout,
		PushConstantRange,
		Sampler,
		SamplerAddressMode,
		SamplerCreateInfo,
		ShaderStageFlags,
	},
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
	},
	graph::{BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, UploadBufferDesc},
	resource::{Image, ImageView, Resource},
};

struct Textures {
	pipeline: Pipeline,
	layout: PipelineLayout,
	image: Image,
	view: ImageView,
	sampler: Sampler,
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
			Format::B8G8R8A8_SRGB,
			&[PushConstantRange::builder()
				.size(std::mem::size_of::<PushConstants>() as _)
				.offset(0)
				.stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
				.build()],
		);

		let (image, view) = load::image(device, include_bytes!("red.webp"), load::ImageFormat::WebP);

		let sampler = unsafe {
			device
				.device()
				.create_sampler(
					&SamplerCreateInfo::builder()
						.mag_filter(Filter::LINEAR)
						.min_filter(Filter::LINEAR)
						.address_mode_u(SamplerAddressMode::REPEAT)
						.address_mode_v(SamplerAddressMode::REPEAT)
						.address_mode_w(SamplerAddressMode::REPEAT),
					None,
				)
				.unwrap()
		};
		let sampler_id = device.base_descriptors().get_sampler(device.device(), sampler);

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
			device.base_descriptors().return_sampler(self.sampler_id);
			self.view.destroy(device);
			self.image.destroy(device);
			device.device().destroy_pipeline_layout(self.layout, None);
			device.device().destroy_pipeline(self.pipeline, None);
		}
	}

	fn render<'frame>(&'frame mut self, frame: &mut Frame<'frame, '_>, input: RenderInput, _: Duration) {
		let mut pass = frame.pass("triangle");
		let (_, write) = pass.output(
			input.image,
			ImageUsage {
				format: input.format,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: ImageViewType::TYPE_2D,
				aspect: ImageAspectFlags::COLOR,
			},
		);

		let (_, index) = pass.output(
			UploadBufferDesc {
				size: std::mem::size_of::<u16>() * 6,
			},
			BufferUsage {
				usages: &[BufferUsageType::IndexBuffer],
			},
		);

		pass.build(move |mut ctx| unsafe {
			let mut index = ctx.write(index);
			index.data.as_mut().copy_from_slice(bytes_of(&[0u16, 1, 3, 1, 2, 3]));

			let view = ctx.write(write);
			cmd::start_rendering_swapchain(ctx.device, ctx.buf, view, input.size);
			ctx.device
				.device()
				.cmd_bind_pipeline(ctx.buf, PipelineBindPoint::GRAPHICS, self.pipeline);

			ctx.device.device().cmd_push_constants(
				ctx.buf,
				self.layout,
				ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
				0,
				bytes_of(&PushConstants {
					image_id: self.view.id.unwrap(),
					sampler_id: self.sampler_id,
					aspect_ratio: input.size.x as f32 / input.size.y as f32,
				}),
			);
			ctx.device.device().cmd_bind_descriptor_sets(
				ctx.buf,
				PipelineBindPoint::GRAPHICS,
				self.layout,
				0,
				&[ctx.device.base_descriptors().set()],
				&[],
			);
			ctx.device
				.device()
				.cmd_bind_index_buffer(ctx.buf, index.buffer, 0, IndexType::UINT16);
			ctx.device.device().cmd_draw_indexed(ctx.buf, 6, 1, 0, 0, 0);

			ctx.device.device().cmd_end_rendering(ctx.buf);
		});
	}
}

fn main() { run::<Textures>() }
