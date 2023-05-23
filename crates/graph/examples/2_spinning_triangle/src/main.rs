use std::time::Duration;

use bytemuck::{bytes_of, NoUninit};
use helpers::{cmd, pipeline, run, vek::mat::repr_simd::row_major::Mat4, App, RenderInput, ShaderStage};
use radiance_graph::{
	ash::vk::{
		Format,
		ImageAspectFlags,
		ImageViewType,
		Pipeline,
		PipelineBindPoint,
		PipelineLayout,
		PushConstantRange,
		ShaderStageFlags,
	},
	device::{descriptor::BufferId, Device},
	graph::{BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, Shader, UploadBufferDesc},
};

const ROTATION_RATE: f32 = 30.0;

struct SpinningTriangle {
	pipeline: Pipeline,
	layout: PipelineLayout,
	deg: f32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	id: BufferId,
	aspect_ratio: f32,
}

impl App for SpinningTriangle {
	const NAME: &'static str = "spinning triangle";

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
				.stage_flags(ShaderStageFlags::VERTEX)
				.build()],
		);

		Self {
			pipeline,
			layout,
			deg: 0.0,
		}
	}

	fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_pipeline_layout(self.layout, None);
			device.device().destroy_pipeline(self.pipeline, None);
		}
	}

	fn render<'frame>(&'frame mut self, frame: &mut Frame<'frame, '_>, input: RenderInput, dt: Duration) {
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
		let (_, storage) = pass.output(
			UploadBufferDesc {
				size: std::mem::size_of::<Mat4<f32>>() as _,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);

		pass.build(move |mut ctx| unsafe {
			let view = ctx.write(write);
			cmd::start_rendering_swapchain(ctx.device, ctx.buf, view, input.size);
			ctx.device
				.device()
				.cmd_bind_pipeline(ctx.buf, PipelineBindPoint::GRAPHICS, self.pipeline);

			let mut storage = ctx.write(storage);
			let rot = Mat4::rotation_z(self.deg.to_radians());
			self.deg += ROTATION_RATE * dt.as_secs_f32();
			storage.data.as_mut().copy_from_slice(bytes_of(&rot.into_row_array()));

			ctx.device.device().cmd_push_constants(
				ctx.buf,
				self.layout,
				ShaderStageFlags::VERTEX,
				0,
				bytes_of(&PushConstants {
					id: storage.id.unwrap(),
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
			ctx.device.device().cmd_draw(ctx.buf, 3, 1, 0, 0);

			ctx.device.device().cmd_end_rendering(ctx.buf);
		});
	}
}

fn main() { run::<SpinningTriangle>() }
