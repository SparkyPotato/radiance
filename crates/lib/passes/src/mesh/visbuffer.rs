use std::io::Write;

use ash::{extensions::ext, vk};
use bytemuck::{bytes_of, NoUninit};
use radiance_asset_runtime::{MeshletPointer, Scene};
use radiance_core::{pipeline::GraphicsPipelineDesc, CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	graph::{
		BufferUsage,
		BufferUsageType,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		ReadId,
		Shader,
		UploadBufferDesc,
		WriteId,
	},
	resource::{ImageView, UploadBufferHandle},
	Result,
};
use radiance_shader_compiler::c_str;
use radiance_util::pipeline::{no_blend, reverse_depth, simple_blend};
use vek::{Mat4, Vec2};

#[derive(Copy, Clone)]
pub struct Camera {
	/// Vertical FOV in radians.
	pub fov: f32,
	pub near: f32,
	/// View matrix (inverse of camera transform).
	pub view: Mat4<f32>,
}

pub struct VisBuffer {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	mesh: ext::MeshShader,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct CameraData {
	view: Mat4<f32>,
	proj: Mat4<f32>,
	view_proj: Mat4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	meshlet_pointers: BufferId,
	camera: BufferId,
	meshlet_count: u32,
}

struct PassIO {
	instances: BufferId,
	meshlet_pointers: BufferId,
	camera_data: CameraData,
	meshlet_count: u32,
	camera: WriteId<UploadBufferHandle>,
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
						.stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT)
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
								c_str!("radiance-passes/mesh/visbuffer/task"),
								vk::ShaderStageFlags::TASK_EXT,
								None,
							)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/mesh/visbuffer/mesh"),
								vk::ShaderStageFlags::MESH_EXT,
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

			Ok(Self {
				pipeline,
				layout,
				mesh: ext::MeshShader::new(device.instance(), device.device()),
			})
		}
	}

	/// Note: `camera` must be setup for reverse Z.
	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, scene: &'pass Scene, camera: Camera, size: Vec2<u32>,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("visbuffer");

		let aspect = size.x as f32 / size.y as f32;
		let proj = infinite_projection(aspect, camera.fov, camera.near);
		let view = camera.view;
		let view_proj = proj * view;

		let camera_data = CameraData { view, proj, view_proj };

		let (_, c) = pass.output(
			UploadBufferDesc {
				size: std::mem::size_of_val(&camera_data) as _,
			},
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::ShaderStorageRead(Shader::Mesh),
				],
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
					instances: scene.instances.inner().inner.id().unwrap(),
					meshlet_pointers: scene.meshlet_pointers.inner().inner.id().unwrap(),
					camera_data,
					meshlet_count: scene.meshlet_pointers.len() as u32 / std::mem::size_of::<MeshletPointer>() as u32,
					camera: c,
					visbuffer: v_w,
					depth: d_w,
				},
			)
		});

		v_r
	}

	fn execute(&self, mut pass: CorePass, io: PassIO) {
		let mut camera = pass.write(io.camera);
		let visbuffer = pass.write(io.visbuffer);
		let depth = pass.write(io.depth);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			camera.data.as_mut().write(bytes_of(&io.camera_data)).unwrap();

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
			let height = visbuffer.size.height as f32;
			dev.cmd_set_viewport(
				buf,
				0,
				&[vk::Viewport {
					x: 0.0,
					y: height,
					width: visbuffer.size.width as f32,
					height: -height,
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
				vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT,
				0,
				bytes_of(&PushConstants {
					instances: io.instances,
					meshlet_pointers: io.meshlet_pointers,
					meshlet_count: io.meshlet_count,
					camera: camera.id.unwrap(),
				}),
			);

			self.mesh.cmd_draw_mesh_tasks(buf, (io.meshlet_count + 63) / 64, 1, 1);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}

fn infinite_projection(aspect: f32, yfov: f32, near: f32) -> Mat4<f32> {
	let h = 1.0 / (yfov / 2.0).tan();
	let w = h / aspect;

	Mat4::new(
		w, 0.0, 0.0, 0.0, //
		0.0, h, 0.0, 0.0, //
		0.0, 0.0, 0.0, near, //
		0.0, 0.0, 1.0, 0.0, //
	)
}
