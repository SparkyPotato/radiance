use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_core::{pipeline::GraphicsPipelineDesc, CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::ImageId,
	graph::{ImageDesc, ImageUsage, ImageUsageFull, ImageUsageType, ReadId, Shader},
	resource::ImageView,
	Result,
};
use radiance_shader_compiler::c_str;
use radiance_util::pipeline::{no_blend, simple_blend};
use vek::Vec2;

pub struct HzbGen {
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	depth: ImageId,
	prev_level: u32,
	dim: Vec2<u32>,
}

struct PassIO {
	depth: ReadId<ImageView>,
	write: ReadId<ImageView>,
	prev_level: u32,
	size: Vec2<u32>,
}

impl HzbGen {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::builder()
						.stage_flags(vk::ShaderStageFlags::FRAGMENT)
						.size(std::mem::size_of::<PushConstants>() as u32)
						.build()]),
				None,
			)?;

			let pipeline = core.graphics_pipeline(
				device,
				&GraphicsPipelineDesc {
					shaders: &[
						core.shaders
							.shader(c_str!("radiance-core/util/screen"), vk::ShaderStageFlags::VERTEX, None)
							.build(),
						core.shaders
							.shader(c_str!("radiance-passes/mesh/hzb"), vk::ShaderStageFlags::FRAGMENT, None)
							.build(),
					],
					blend: &simple_blend(&[no_blend()]),
					layout,
					color_attachments: &[],
					depth_attachment: vk::Format::D32_SFLOAT,
					..Default::default()
				},
			)?;

			Ok(Self { layout, pipeline })
		}
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, depth: ReadId<ImageView>, mut size: Vec2<u32>,
	) -> ReadId<ImageView> {
		let levels = size.x.max(size.y).ilog2();
		let read = self.init_pass(frame, depth, size, levels);

		for i in 1..levels {
			if size.x > 1 {
				size.x /= 2
			}
			if size.y > 1 {
				size.y /= 2
			}

			let mut pass = frame.pass("generate hzb");
			pass.input(
				read,
				ImageUsageFull {
					format: vk::Format::D32_SFLOAT,
					usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
					view_type: vk::ImageViewType::TYPE_2D,
					aspect: vk::ImageAspectFlags::DEPTH,
					levels: (i - 1)..i,
					layers: 0..1,
				},
			);
			pass.input(
				read,
				ImageUsageFull {
					format: vk::Format::D32_SFLOAT,
					usages: &[ImageUsageType::DepthStencilAttachmentWrite],
					view_type: vk::ImageViewType::TYPE_2D,
					aspect: vk::ImageAspectFlags::DEPTH,
					levels: i..(i + 1),
					layers: 0..1,
				},
			);

			pass.build(move |pass| {
				self.execute(
					pass,
					PassIO {
						depth,
						write: read,
						prev_level: i - 1,
						size,
					},
				)
			});
		}

		read
	}

	fn init_pass<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, depth: ReadId<ImageView>, mut size: Vec2<u32>, levels: u32,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("generate hzb");
		pass.input(
			depth,
			ImageUsage {
				format: vk::Format::D32_SFLOAT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::DEPTH,
			},
		);
		let (read, write) = pass.output(
			ImageDesc {
				size: vk::Extent3D {
					width: size.x / 2,
					height: size.y / 2,
					depth: 1,
				},
				levels,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
			},
			ImageUsageFull {
				format: vk::Format::D32_SFLOAT,
				usages: &[ImageUsageType::DepthStencilAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::DEPTH,
				levels: 0..1,
				layers: 0..1,
			},
		);
		let write = write.to_read();
		pass.build(move |pass| {
			self.execute(
				pass,
				PassIO {
					depth,
					write,
					prev_level: 0,
					size,
				},
			)
		});
		read
	}

	fn execute(&self, mut pass: CorePass, mut io: PassIO) {
		let depth = pass.read(io.depth);
		let write = pass.read(io.write);
		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let prev_size = io.size;
			if io.size.x > 1 {
				io.size.x /= 2
			}
			if io.size.y > 1 {
				io.size.y /= 2
			}

			let area = vk::Rect2D::builder()
				.extent(vk::Extent2D {
					width: io.size.x,
					height: io.size.y,
				})
				.build();
			dev.cmd_begin_rendering(
				buf,
				&vk::RenderingInfo::builder()
					.render_area(area)
					.layer_count(1)
					.depth_attachment(
						&vk::RenderingAttachmentInfo::builder()
							.image_view(write.view)
							.image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
							.load_op(vk::AttachmentLoadOp::CLEAR)
							.clear_value(vk::ClearValue {
								depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
							})
							.store_op(vk::AttachmentStoreOp::DONT_CARE),
					),
			);
			let height = io.size.y as f32;
			dev.cmd_set_viewport(
				buf,
				0,
				&[vk::Viewport {
					x: 0.0,
					y: height,
					width: io.size.x as f32,
					height: -height,
					min_depth: 0.0,
					max_depth: 1.0,
				}],
			);
			dev.cmd_set_scissor(buf, 0, &[area]);
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
					depth: depth.id.unwrap(),
					prev_level: io.prev_level,
					dim: prev_size,
				}),
			);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
			dev.cmd_draw(buf, 3, 1, 0, 0);
			dev.cmd_end_rendering(buf);
		}
	}

	pub fn destroy(self, device: &CoreDevice) {
		unsafe {
			device.device().destroy_pipeline(self.pipeline, None);
			device.device().destroy_pipeline_layout(self.layout, None);
		}
	}
}
