use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{descriptor::StorageImageId, Device, GraphicsPipelineDesc, Pipeline, ShaderInfo},
	graph::{BufferUsage, BufferUsageType, Frame, ImageDesc, ImageUsage, ImageUsageType, PassContext, Res, Shader},
	resource::{GpuPtr, ImageView, Subresource},
	util::pipeline::{no_blend, no_cull, simple_blend},
	Result,
};

use crate::{
	asset::scene::GpuInstance,
	mesh::{CameraData, DebugResId, RenderOutput},
};

#[derive(Copy, Clone)]
pub enum DebugVis {
	Triangles,
	Meshlets,
	Overdraw(u32, u32),
	HwSw,
	Normals,
	HzbMip,
}

impl DebugVis {
	pub fn requires_debug_info(self) -> bool { matches!(self, Self::Overdraw(..) | Self::HwSw) }
}

pub struct DebugMesh {
	triangles: Pipeline,
	meshlets: Pipeline,
	overdraw: Pipeline,
	hwsw: Pipeline,
	normals: Pipeline,
	hzb_mip: Pipeline,
	layout: vk::PipelineLayout,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	early: GpuPtr<u8>,
	late: GpuPtr<u8>,
	visbuffer: StorageImageId,
	debug: Option<DebugResId>,
	bottom: u32,
	top: u32,
	_pad: u32,
}

impl DebugMesh {
	fn pipeline(device: &Device, layout: vk::PipelineLayout, shader: &'static str) -> Result<Pipeline> {
		device.graphics_pipeline(GraphicsPipelineDesc {
			layout,
			shaders: &[
				ShaderInfo {
					shader: "graph.util.screen",
					..Default::default()
				},
				ShaderInfo {
					shader,
					spec: &["passes.mesh.debug"],
				},
			],
			raster: no_cull(),
			blend: simple_blend(&[no_blend()]),
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

			Ok(Self {
				layout,
				triangles: Self::pipeline(device, layout, "passes.debug.triangles")?,
				meshlets: Self::pipeline(device, layout, "passes.debug.meshlets")?,
				overdraw: Self::pipeline(device, layout, "passes.debug.overdraw")?,
				hwsw: Self::pipeline(device, layout, "passes.debug.hwsw")?,
				normals: Self::pipeline(device, layout, "passes.debug.normals")?,
				hzb_mip: Self::pipeline(device, layout, "passes.debug.hzb_mip")?,
			})
		}
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, vis: DebugVis, output: RenderOutput,
	) -> Res<ImageView> {
		let mut pass = frame.pass("debug mesh");
		let usage = BufferUsage {
			usages: &[BufferUsageType::ShaderStorageRead(Shader::Fragment)],
		};
		pass.reference(output.camera, usage);
		pass.reference(output.early, usage);
		pass.reference(output.late, usage);

		let usage = ImageUsage {
			format: vk::Format::UNDEFINED,
			usages: &[ImageUsageType::ShaderStorageRead(Shader::Fragment)],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};
		pass.reference(output.visbuffer, usage);
		let desc = pass.desc(output.visbuffer);
		let out = pass.resource(
			ImageDesc {
				format: vk::Format::R8G8B8A8_SRGB,
				..desc
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		if let Some(d) = output.debug {
			pass.reference(d.overdraw, usage);
			pass.reference(d.hwsw, usage);
		}

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
					DebugVis::Triangles => self.triangles.get(),
					DebugVis::Meshlets => self.meshlets.get(),
					DebugVis::Overdraw(..) => self.overdraw.get(),
					DebugVis::HwSw => self.hwsw.get(),
					DebugVis::Normals => self.normals.get(),
					DebugVis::HzbMip => self.hzb_mip.get(),
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
			let (bottom, top) = match vis {
				DebugVis::Overdraw(bottom, top) => (bottom, top),
				_ => (0, 0),
			};
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::FRAGMENT,
				0,
				bytes_of(&PushConstants {
					instances: output.instances,
					camera: pass.get(output.camera).ptr(),
					early: pass.get(output.early).ptr(),
					late: pass.get(output.late).ptr(),
					visbuffer: visbuffer.storage_id.unwrap(),
					debug: output.debug.map(|d| d.get(&mut pass)),
					bottom,
					top,
					_pad: 0,
				}),
			);

			dev.cmd_draw(buf, 3, 1, 0, 0);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.triangles.destroy();
		self.meshlets.destroy();
		self.overdraw.destroy();
		self.hwsw.destroy();
		self.normals.destroy();
		self.hzb_mip.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
