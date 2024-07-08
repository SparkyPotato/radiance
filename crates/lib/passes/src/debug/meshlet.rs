use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{descriptor::ImageId, Device},
	graph::{Frame, ImageUsage, ImageUsageType, PassContext, Res, Shader},
	resource::{ImageView, Subresource},
	util::pipeline::{no_blend, no_cull, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::c_str;

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
	pub fn new(device: &Device) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::FRAGMENT)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?;

			let pipeline = device.graphics_pipeline(&GraphicsPipelineDesc {
				layout,
				shaders: &[
					device.shader(c_str!("radiance-graph/util/screen"), vk::ShaderStageFlags::VERTEX, None),
					device.shader(
						c_str!("radiance-passes/debug/meshlet"),
						vk::ShaderStageFlags::FRAGMENT,
						None,
					),
				],
				color_attachments: &[vk::Format::R8G8B8A8_SRGB],
				raster: &no_cull(),
				blend: &simple_blend(&[no_blend()]),
				..Default::default()
			})?;

			Ok(Self { layout, pipeline })
		}
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, visbuffer: Res<ImageView>) -> Res<ImageView> {
		let mut pass = frame.pass("debug meshlets");
		pass.reference(
			visbuffer,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		let output = pass.resource(
			visbuffer,
			ImageUsage {
				format: vk::Format::R8G8B8A8_SRGB, // TODO: fix
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		pass.build(move |ctx| self.execute(ctx, visbuffer, output));

		output
	}

	fn execute(&self, mut pass: PassContext, visbuffer: Res<ImageView>, out: Res<ImageView>) {
		let visbuffer = pass.get(visbuffer);
		let out = pass.get(out);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let area = vk::Rect2D::default().extent(vk::Extent2D {
				width: visbuffer.size.width,
				height: visbuffer.size.height,
			});
			dev.cmd_begin_rendering(
				buf,
				&vk::RenderingInfo::default()
					.render_area(area)
					.layer_count(1)
					.color_attachments(&[vk::RenderingAttachmentInfo::default()
						.image_view(out.view)
						.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue {
								float32: [0.0, 0.0, 0.0, 1.0],
							},
						})
						.store_op(vk::AttachmentStoreOp::STORE)]),
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

	pub unsafe fn destroy(self, device: &Device) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
