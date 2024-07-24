use std::io::Write;

use ash::{ext, vk};
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{descriptor::BufferId, Device},
	graph::{
		self,
		BufferUsage,
		BufferUsageType,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Res,
		Shader,
	},
	resource::{BufferHandle, ImageView, Subresource},
	util::pipeline::{no_blend, reverse_depth, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::{Mat4, Vec2};

use crate::asset::{rref::RRef, scene::Scene};

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Camera {
	/// Vertical FOV in radians.
	pub fov: f32,
	pub near: f32,
	/// View matrix (inverse of camera transform).
	pub view: Mat4<f32>,
}

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub camera: Camera,
	pub cull_camera: Option<Camera>,
	pub size: Vec2<u32>,
}

pub struct VisBuffer {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	mesh: ext::mesh_shader::Device,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct CameraData {
	view: Mat4<f32>,
	proj: Mat4<f32>,
	view_proj: Mat4<f32>,
	cot_fov: f32,
	_pad: [f32; 15],
}

impl CameraData {
	fn new(aspect: f32, camera: Camera) -> Self {
		let proj = infinite_projection(aspect, camera.fov, camera.near);
		let view = camera.view;
		let view_proj = proj * view;

		Self {
			view,
			proj,
			view_proj,
			cot_fov: (camera.fov / 2.0).tan().recip(),
			_pad: [0.0; 15],
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	meshlet_pointers: BufferId,
	camera: BufferId,
	meshlet_count: u32,
	resolution: u32,
}

struct PassIO {
	instances: BufferId,
	meshlet_pointers: BufferId,
	cull_camera: CameraData,
	draw_camera: CameraData,
	meshlet_count: u32,
	resolution: u32,
	camera: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	depth: Res<ImageView>,
}

impl VisBuffer {
	fn pipeline(device: &Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
		device.graphics_pipeline(&GraphicsPipelineDesc {
			shaders: &[
				device.shader(
					c_str!("radiance-passes/mesh/visbuffer/task"),
					vk::ShaderStageFlags::TASK_EXT,
					None,
				),
				device.shader(
					c_str!("radiance-passes/mesh/visbuffer/mesh"),
					vk::ShaderStageFlags::MESH_EXT,
					None,
				),
				device.shader(
					c_str!("radiance-passes/mesh/visbuffer/pixel"),
					vk::ShaderStageFlags::FRAGMENT,
					None,
				),
			],
			depth: &reverse_depth(),
			blend: &simple_blend(&[no_blend()]),
			layout,
			color_attachments: &[vk::Format::R32_UINT],
			depth_attachment: vk::Format::D32_SFLOAT,
			..Default::default()
		})
	}

	pub fn new(device: &Device) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?;

			Ok(Self {
				pipeline: Self::pipeline(device, layout)?,
				layout,
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let mut pass = frame.pass("visbuffer");

		let aspect = info.size.x as f32 / info.size.y as f32;
		let draw_camera = CameraData::new(aspect, info.camera);
		let cull_camera = info
			.cull_camera
			.map(|c| CameraData::new(aspect, c))
			.unwrap_or(draw_camera);

		let c = pass.resource(
			graph::BufferDesc {
				size: (std::mem::size_of::<CameraData>() * 2) as _,
				upload: true,
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
				width: info.size.x,
				height: info.size.y,
				depth: 1,
			},
			format: vk::Format::R32_UINT,
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
		};
		let visbuffer = pass.resource(
			desc,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		let depth = pass.resource(
			ImageDesc {
				format: vk::Format::D32_SFLOAT,
				..desc
			},
			ImageUsage {
				format: vk::Format::D32_SFLOAT,
				usages: &[ImageUsageType::DepthStencilAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource {
					aspect: vk::ImageAspectFlags::DEPTH,
					..Default::default()
				},
			},
		);

		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					instances: info.scene.instances(),
					meshlet_pointers: info.scene.meshlet_pointers(),
					cull_camera,
					draw_camera,
					meshlet_count: info.scene.meshlet_pointer_count(),
					resolution: info.size.x.max(info.size.y),
					camera: c,
					visbuffer,
					depth,
				},
			)
		});

		visbuffer
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let mut camera = pass.get(io.camera);
		let visbuffer = pass.get(io.visbuffer);
		let depth = pass.get(io.depth);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let mut writer = camera.data.as_mut();
			writer.write(bytes_of(&io.cull_camera)).unwrap();
			writer.write(bytes_of(&io.draw_camera)).unwrap();

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
						.image_view(visbuffer.view)
						.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
						})
						.store_op(vk::AttachmentStoreOp::STORE)])
					.depth_attachment(
						&vk::RenderingAttachmentInfo::default()
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
					camera: camera.id.unwrap(),
					meshlet_count: io.meshlet_count,
					resolution: io.resolution,
				}),
			);

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
			self.mesh.cmd_draw_mesh_tasks(buf, (io.meshlet_count + 63) / 64, 1, 1);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}

pub fn infinite_projection(aspect: f32, yfov: f32, near: f32) -> Mat4<f32> {
	let h = 1.0 / (yfov / 2.0).tan();
	let w = h / aspect;

	Mat4::new(
		w, 0.0, 0.0, 0.0, //
		0.0, h, 0.0, 0.0, //
		0.0, 0.0, 0.0, near, //
		0.0, 0.0, 1.0, 0.0, //
	)
}
