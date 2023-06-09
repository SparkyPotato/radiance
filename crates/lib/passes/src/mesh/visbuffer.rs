use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_asset_runtime::{MeshletPointer, Scene};
use radiance_core::{pipeline::GraphicsPipelineDesc, CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	graph::{BufferUsage, BufferUsageType, ImageDesc, ImageUsage, ImageUsageType, ReadId, WriteId},
	resource::ImageView,
	Result,
};
use radiance_shader_compiler::c_str;
use radiance_util::pipeline::{no_blend, reverse_depth, simple_blend};
use vek::Vec2;

use crate::mesh::cull::{Command, CullOutput};

pub struct VisBuffer {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	meshlets: BufferId,
	instances: BufferId,
	meshlet_pointers: BufferId,
	vertices: BufferId,
	camera: BufferId,
}

struct PassIO {
	vertices: BufferId,
	indices: vk::Buffer,
	meshlets: BufferId,
	instances: BufferId,
	meshlet_pointers: BufferId,
	total_meshlet_count: u32,
	cull: CullOutput,
	visbuffer: WriteId<ImageView>,
	depth: WriteId<ImageView>,
}

impl VisBuffer {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::builder()
						.stage_flags(vk::ShaderStageFlags::VERTEX)
						.size(std::mem::size_of::<PushConstants>() as u32)
						.build()]),
				None,
			)?;

			let pipeline = core.graphics_pipeline(
				device,
				&GraphicsPipelineDesc {
					shaders: &[
						core.shaders
							.shader(
								c_str!("radiance-passes/mesh/visbuffer/vertex"),
								vk::ShaderStageFlags::VERTEX,
								None,
							)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/mesh/visbuffer/pixel"),
								vk::ShaderStageFlags::FRAGMENT,
								None,
							)
							.build(),
					],
					depth: &reverse_depth(),
					blend: &simple_blend(&[no_blend()]),
					layout,
					color_attachments: &[vk::Format::R32_UINT],
					depth_attachment: vk::Format::D32_SFLOAT,
					..Default::default()
				},
			)?;

			Ok(Self { pipeline, layout })
		}
	}

	/// Note: `camera_viewproj` must be setup for reverse Z.
	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, scene: &'pass Scene, cull: CullOutput, size: Vec2<u32>,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("visbuffer");
		pass.input(
			cull.commands,
			BufferUsage {
				usages: &[BufferUsageType::IndirectBuffer],
			},
		);
		pass.input(
			cull.draw_count,
			BufferUsage {
				usages: &[BufferUsageType::IndirectBuffer],
			},
		);
		let desc = ImageDesc {
			size: vk::Extent3D {
				width: size.x,
				height: size.y,
				depth: 1,
			},
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
		};
		let (v_r, v_w) = pass.output(
			desc,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);
		let (_, d_w) = pass.output(
			desc,
			ImageUsage {
				format: vk::Format::D32_SFLOAT,
				usages: &[ImageUsageType::DepthStencilAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::DEPTH,
			},
		);

		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					meshlets: scene.meshlets.inner().inner.id().unwrap(),
					instances: scene.instances.inner().inner.id().unwrap(),
					meshlet_pointers: scene.meshlet_pointers.inner().inner.id().unwrap(),
					vertices: scene.vertices.inner().inner.id().unwrap(),
					indices: scene.indices.inner().inner.inner(),
					total_meshlet_count: scene.meshlet_pointers.len() as u32
						/ std::mem::size_of::<MeshletPointer>() as u32,
					cull,
					visbuffer: v_w,
					depth: d_w,
				},
			)
		});

		v_r
	}

	fn execute(&self, mut pass: CorePass, io: PassIO) {
		let commands = pass.read(io.cull.commands);
		let draw_count = pass.read(io.cull.draw_count);
		let camera = pass.read(io.cull.camera);
		let visbuffer = pass.write(io.visbuffer);
		let depth = pass.write(io.depth);

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
						.image_view(visbuffer.view)
						.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
						})
						.store_op(vk::AttachmentStoreOp::STORE)
						.build()])
					.depth_attachment(
						&vk::RenderingAttachmentInfo::builder()
							.image_view(depth.view)
							.image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
							.load_op(vk::AttachmentLoadOp::CLEAR)
							.clear_value(vk::ClearValue {
								depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
							})
							.store_op(vk::AttachmentStoreOp::DONT_CARE),
					),
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
				vk::ShaderStageFlags::VERTEX,
				0,
				bytes_of(&PushConstants {
					meshlets: io.meshlets,
					instances: io.instances,
					meshlet_pointers: io.meshlet_pointers,
					vertices: io.vertices,
					camera: camera.id.unwrap(),
				}),
			);
			dev.cmd_bind_index_buffer(buf, io.indices, 0, vk::IndexType::UINT8_EXT);

			dev.cmd_draw_indexed_indirect_count(
				buf,
				commands.buffer,
				0,
				draw_count.buffer,
				0,
				io.total_meshlet_count,
				std::mem::size_of::<Command>() as _,
			);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
