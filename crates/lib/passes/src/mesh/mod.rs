use ash::{ext, vk};
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{descriptor::BufferId, Device},
	graph::{BufferUsage, BufferUsageType, Frame, ImageDesc, ImageUsage, ImageUsageType, PassContext, Res, Shader},
	resource::{BufferHandle, Image, ImageView, Resource, Subresource},
	util::pipeline::{no_blend, reverse_depth, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::{Mat4, Vec2, Vec4};

use crate::{
	asset::{rref::RRef, scene::Scene},
	mesh::{bvh::BvhCull, hzb::HzbGen, instance::InstanceCull, meshlet::MeshletCull, setup::Setup},
};

mod bvh;
mod hzb;
mod instance;
mod meshlet;
mod setup;

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
	pub size: Vec2<u32>,
}

pub struct VisBuffer {
	setup: Setup,
	early_instance_cull: InstanceCull,
	late_instance_cull: InstanceCull,
	early_bvh_cull: BvhCull,
	late_bvh_cull: BvhCull,
	early_meshlet_cull: MeshletCull,
	late_meshlet_cull: MeshletCull,
	hzb_gen: HzbGen,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	mesh: ext::mesh_shader::Device,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct CameraData {
	view: Mat4<f32>,
	view_proj: Mat4<f32>,
	w: f32,
	h: f32,
	near: f32,
	_pad: f32,
	frustum: Vec4<f32>,
}

fn normalize_plane(p: Vec4<f32>) -> Vec4<f32> { p / p.xyz().magnitude() }

impl CameraData {
	fn new(aspect: f32, camera: Camera) -> Self {
		let h = (camera.fov / 2.0).tan().recip();
		let w = h / aspect;
		let near = camera.near;
		let proj = Mat4::new(
			w, 0.0, 0.0, 0.0, //
			0.0, h, 0.0, 0.0, //
			0.0, 0.0, 0.0, near, //
			0.0, 0.0, 1.0, 0.0, //
		);
		let view = camera.view;
		let view_proj = proj * view;

		let pt = proj.transposed();
		let px = normalize_plane(pt.cols[3] + pt.cols[0]);
		let py = normalize_plane(pt.cols[3] + pt.cols[1]);
		let frustum = Vec4::new(px.x, px.z, py.y, py.z);

		Self {
			view,
			view_proj,
			w,
			h,
			near,
			_pad: 0.0,
			frustum,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	camera: BufferId,
	read: BufferId,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: BufferId,
	camera: Res<BufferHandle>,
	read: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	depth: Res<ImageView>,
}

impl VisBuffer {
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
			let pipeline = device.graphics_pipeline(&GraphicsPipelineDesc {
				shaders: &[
					device.shader(
						c_str!("radiance-passes/mesh/mesh"),
						vk::ShaderStageFlags::MESH_EXT,
						None,
					),
					device.shader(
						c_str!("radiance-passes/mesh/pixel"),
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
			})?;

			Ok(Self {
				setup: Setup::new(),
				early_instance_cull: InstanceCull::new(device, true)?,
				late_instance_cull: InstanceCull::new(device, false)?,
				early_bvh_cull: BvhCull::new(device, true)?,
				late_bvh_cull: BvhCull::new(device, false)?,
				early_meshlet_cull: MeshletCull::new(device, true)?,
				late_meshlet_cull: MeshletCull::new(device, false)?,
				hzb_gen: HzbGen::new(device)?,
				layout,
				pipeline,
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let res = self.setup.run(frame, &info, self.hzb_gen.sampler());
		let this: &Self = self;

		this.early_instance_cull.run(frame, &info, &res);
		this.early_bvh_cull.run(frame, &info, &res);
		this.early_meshlet_cull.run(frame, &info, &res);

		let mut pass = frame.pass("visbuffer early");
		let camera = res.camera_mesh(&mut pass);
		let read = res.input_mesh(&mut pass, res.meshlet_render_lists[0]);
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
		let visbuffer_usage = ImageUsage {
			format: vk::Format::R32_UINT,
			usages: &[ImageUsageType::ColorAttachmentWrite],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};
		let depth_usage = ImageUsage {
			format: vk::Format::D32_SFLOAT,
			usages: &[ImageUsageType::DepthStencilAttachmentWrite],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource {
				aspect: vk::ImageAspectFlags::DEPTH,
				..Default::default()
			},
		};
		let visbuffer = pass.resource(desc, visbuffer_usage);
		let depth = pass.resource(
			ImageDesc {
				format: vk::Format::D32_SFLOAT,
				..desc
			},
			depth_usage,
		);
		let mut io = PassIO {
			instances: info.scene.instances(),
			camera,
			read,
			visbuffer,
			depth,
		};
		pass.build(move |ctx| this.execute(ctx, io, true));

		this.hzb_gen.run(frame, depth, res.hzb);
		this.late_instance_cull.run(frame, &info, &res);
		this.late_bvh_cull.run(frame, &info, &res);
		this.late_meshlet_cull.run(frame, &info, &res);

		let mut pass = frame.pass("visbuffer late");
		res.camera_mesh(&mut pass);

		pass.reference(visbuffer, visbuffer_usage);
		pass.reference(depth, depth_usage);
		io.read = res.input_mesh(&mut pass, res.meshlet_render_lists[1]);
		pass.build(move |ctx| this.execute(ctx, io, false));

		this.hzb_gen.run(frame, depth, res.hzb);

		visbuffer
	}

	fn execute(&self, mut pass: PassContext, io: PassIO, is_early: bool) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let read = pass.get(io.read);
		let visbuffer = pass.get(io.visbuffer);
		let depth = pass.get(io.depth);
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
						.image_view(visbuffer.view)
						.image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
						.load_op(if is_early {
							vk::AttachmentLoadOp::CLEAR
						} else {
							vk::AttachmentLoadOp::LOAD
						})
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue {
								uint32: [u32::MAX, 0, 0, 0],
							},
						})
						.store_op(vk::AttachmentStoreOp::STORE)])
					.depth_attachment(
						&vk::RenderingAttachmentInfo::default()
							.image_view(depth.view)
							.image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
							.load_op(if is_early {
								vk::AttachmentLoadOp::CLEAR
							} else {
								vk::AttachmentLoadOp::LOAD
							})
							.clear_value(vk::ClearValue {
								depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
							})
							.store_op(vk::AttachmentStoreOp::STORE),
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
					camera: pass.get(io.camera).id.unwrap(),
					read: read.id.unwrap(),
				}),
			);

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
			self.mesh
				.cmd_draw_mesh_tasks_indirect(buf, read.buffer, 0, 1, std::mem::size_of::<u32>() as u32 * 3);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.setup.destroy(device);
		self.early_instance_cull.destroy(device);
		self.late_instance_cull.destroy(device);
		self.early_bvh_cull.destroy(device);
		self.late_bvh_cull.destroy(device);
		self.early_meshlet_cull.destroy(device);
		self.late_meshlet_cull.destroy(device);
		self.hzb_gen.destroy(device);
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
