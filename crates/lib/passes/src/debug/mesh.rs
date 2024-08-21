use std::ffi::CStr;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{BufferId, ImageId},
		Device,
	},
	graph::{Frame, ImageUsage, ImageUsageType, PassContext, Res, Shader},
	resource::{ImageView, Subresource},
	util::pipeline::{no_blend, no_cull, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::c_str;

use crate::mesh::RenderOutput;

#[derive(Copy, Clone)]
pub enum DebugVis {
	Triangles,
	Meshlets,
}

pub struct DebugMesh {
	triangles: vk::Pipeline,
	meshlets: vk::Pipeline,
	layout: vk::PipelineLayout,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	visbuffer: ImageId,
	early: BufferId,
	late: BufferId,
}

impl DebugMesh {
	fn pipeline(device: &Device, layout: vk::PipelineLayout, name: &CStr) -> Result<vk::Pipeline> {
		device.graphics_pipeline(&GraphicsPipelineDesc {
			layout,
			shaders: &[
				device.shader(c_str!("radiance-graph/util/screen"), vk::ShaderStageFlags::VERTEX, None),
				device.shader(name, vk::ShaderStageFlags::FRAGMENT, None),
			],
			raster: &no_cull(),
			blend: &simple_blend(&[no_blend()]),
			color_attachments: &[vk::Format::R8G8B8A8_SRGB],
			..Default::default()
		})
	}

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

			let triangles = Self::pipeline(device, layout, c_str!("radiance-passes/debug/triangles"))?;
			let meshlets = Self::pipeline(device, layout, c_str!("radiance-passes/debug/meshlets"))?;

			Ok(Self {
				layout,
				triangles,
				meshlets,
			})
		}
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, vis: DebugVis, output: RenderOutput,
	) -> Res<ImageView> {
		let mut pass = frame.pass("debug meshlets");
		pass.reference(
			output.visbuffer,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		let out = pass.resource(
			output.visbuffer,
			ImageUsage {
				format: vk::Format::R8G8B8A8_SRGB, // TODO: fix
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		pass.build(move |ctx| self.execute(ctx, vis, output, out));

		out
	}

	fn execute(&self, mut pass: PassContext, vis: DebugVis, output: RenderOutput, out: Res<ImageView>) {
		let visbuffer = pass.get(output.visbuffer);
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
			dev.cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				match vis {
					DebugVis::Triangles => self.triangles,
					DebugVis::Meshlets => self.meshlets,
				},
			);
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
					early: pass.get(output.early).id.unwrap(),
					late: pass.get(output.late).id.unwrap(),
				}),
			);

			dev.cmd_draw(buf, 3, 1, 0, 0);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		device.device().destroy_pipeline(self.triangles, None);
		device.device().destroy_pipeline(self.meshlets, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
