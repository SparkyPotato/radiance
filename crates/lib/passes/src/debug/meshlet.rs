use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_core::{pipeline::GraphicsPipelineDesc, CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::ImageId,
	graph::{ImageUsage, ImageUsageType, ReadId, Shader, WriteId},
	resource::ImageView,
	Result,
};
use radiance_shader_compiler::c_str;
use radiance_util::pipeline::{no_blend, no_cull, simple_blend};

pub struct DebugMeshlets {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	visbuffer: ImageId,
}

impl DebugMeshlets {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::builder()
						.stage_flags(vk::ShaderStageFlags::FRAGMENT)
						.size(std::mem::size_of::<PushConstants>() as u32)
						.build()]),
				None,
			)?;

			let pipeline = core.graphics_pipeline(
				device,
				&GraphicsPipelineDesc {
					layout,
					shaders: &[
						core.shaders
							.shader(c_str!("radiance-core/util/screen"), vk::ShaderStageFlags::VERTEX, None)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/debug/meshlet"),
								vk::ShaderStageFlags::FRAGMENT,
								None,
							)
							.build(),
					],
					color_attachments: &[vk::Format::R8G8B8A8_UNORM],
					raster: &no_cull(),
					blend: &simple_blend(&[no_blend()]),
					..Default::default()
				},
			)?;

			Ok(Self { layout, pipeline })
		}
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, visbuffer: ReadId<ImageView>,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("debug meshlets");
		pass.input(
			visbuffer,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);
		let (ret, output) = pass.output(
			visbuffer,
			ImageUsage {
				format: vk::Format::R8G8B8A8_UNORM, // TODO: fix
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);

		pass.build(move |ctx| self.execute(ctx, visbuffer, output));

		ret
	}

	fn execute(&self, mut pass: CorePass, visbuffer: ReadId<ImageView>, out: WriteId<ImageView>) {
		let visbuffer = pass.read(visbuffer);
		let out = pass.write(out);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let area = vk::Rect2D::builder()
				.extent(vk::Extent2D {
					width: visbuffer.size.width,
					height: visbuffer.size.height,
				})
				.build();
			dev.cmd_begin_rendering(
				buf,
				&vk::RenderingInfo::builder()
					.render_area(area)
					.layer_count(1)
					.color_attachments(&[vk::RenderingAttachmentInfo::builder()
						.image_view(out.view)
						.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue {
								float32: [0.0, 0.0, 0.0, 1.0],
							},
						})
						.store_op(vk::AttachmentStoreOp::STORE)
						.build()]),
			);
			dev.cmd_set_viewport(
				buf,
				0,
				&[vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: visbuffer.size.width as f32,
					height: visbuffer.size.height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				}],
			);
			dev.cmd_set_scissor(buf, 0, &[area]);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::FRAGMENT,
				0,
				bytes_of(&PushConstants {
					visbuffer: visbuffer.id.unwrap(),
				}),
			);

			dev.cmd_draw(buf, 3, 1, 0, 0);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}

