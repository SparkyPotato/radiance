use std::io::Write;

use ash::{ext, vk};
use bytemuck::{bytes_of, cast_slice, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
	},
	graph::{
		self,
		util::ByteReader,
		BufferUsage,
		BufferUsageType,
		ExternalBuffer,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
		Shader,
	},
	resource::{BufferDesc, BufferHandle, ImageView, Subresource},
	util::{
		persistent::PersistentBuffer,
		pipeline::{no_blend, reverse_depth, simple_blend, GraphicsPipelineDesc},
	},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::{Mat4, Vec2};

use crate::{
	asset::{rref::RRef, scene::Scene},
	mesh::hzb::HzbGen,
};

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
	persistent: Option<Persistent>,
	hzb_gen: HzbGen,
	pre_cull_pipeline: vk::Pipeline,
	cull_pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	mesh: ext::mesh_shader::Device,
}

struct Persistent {
	dispatches: PersistentBuffer,
	scene: RRef<Scene>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct CameraData {
	view: Mat4<f32>,
	view_proj: Mat4<f32>,
	w: f32,
	h: f32,
	near: f32,
	_pad: [f32; 13],
}

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

		Self {
			view,
			view_proj,
			w,
			h,
			near,
			_pad: [0.0; 13],
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
	width: u32,
	height: u32,
	curr_dispatch: BufferId,
	next_dispatch: BufferId,
	hzb_sampler: SamplerId,
	hzb: Option<ImageId>,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: BufferId,
	meshlet_pointers: BufferId,
	camera_data: CameraData,
	meshlet_count: u32,
	resolution: Vec2<u32>,
	curr_dispatch: Res<BufferHandle>,
	next_dispatch: Res<BufferHandle>,
	camera: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	depth: Res<ImageView>,
	hzb: Option<Res<ImageView>>,
}

impl VisBuffer {
	fn pipeline(device: &Device, layout: vk::PipelineLayout, pre_cull: bool) -> Result<vk::Pipeline> {
		device.graphics_pipeline(&GraphicsPipelineDesc {
			shaders: &[
				device.shader(
					if pre_cull {
						c_str!("radiance-passes/mesh/visbuffer/pre_cull")
					} else {
						c_str!("radiance-passes/mesh/visbuffer/cull")
					},
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
				persistent: None,
				hzb_gen: HzbGen::new(device)?,
				pre_cull_pipeline: Self::pipeline(device, layout, true)?,
				cull_pipeline: Self::pipeline(device, layout, false)?,
				layout,
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let (curr, next) = self.init_dispatch_buffer(frame, &info.scene);

		let mut pass = frame.pass("visbuffer pre-cull");
		pass.reference(
			curr,
			BufferUsage {
				usages: &[
					BufferUsageType::IndirectBuffer,
					BufferUsageType::ShaderStorageRead(Shader::Task),
				],
			},
		);
		pass.reference(
			next,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageWrite(Shader::Task)],
			},
		);

		let aspect = info.size.x as f32 / info.size.y as f32;
		let camera = CameraData::new(aspect, info.camera);
		let c = pass.resource(
			graph::BufferDesc {
				size: std::mem::size_of::<CameraData>() as _,
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

		let mut io = PassIO {
			instances: info.scene.instances(),
			meshlet_pointers: info.scene.meshlet_pointers(),
			camera_data: camera,
			meshlet_count: info.scene.meshlet_pointer_count(),
			resolution: info.size,
			curr_dispatch: curr,
			next_dispatch: next,
			camera: c,
			visbuffer,
			depth,
			hzb: None,
		};
		let this: &Self = self;
		pass.build(move |ctx| this.execute(ctx, io));

		let hzb = this.hzb_gen.run(frame, depth);

		let mut pass = frame.pass("visbuffer cull");
		pass.reference(curr, BufferUsage { usages: &[] });
		pass.reference(next, BufferUsage { usages: &[] });
		pass.reference(c, BufferUsage { usages: &[] });
		pass.reference(
			visbuffer,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		pass.reference(
			depth,
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
		pass.reference(
			hzb,
			ImageUsage {
				format: vk::Format::R32_SFLOAT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		io.hzb = Some(hzb);
		pass.build(move |ctx| this.execute(ctx, io));

		visbuffer
	}

	fn init_dispatch_buffer(&mut self, frame: &mut Frame, s: &RRef<Scene>) -> (Res<BufferHandle>, Res<BufferHandle>) {
		match &mut self.persistent {
			Some(Persistent { dispatches, scene }) => {
				if scene.ptr_eq(s) {
					let (curr, next) = dispatches.next();
					let mut pass = frame.pass("clear visbuffer dispatch");
					let curr = pass.resource(curr, BufferUsage { usages: &[] });
					let next = Self::zero_next(pass, s, next);
					(curr, next)
				} else {
					let (b, curr, next) = Self::init_inner(frame, s);
					frame.delete(std::mem::replace(dispatches, b));
					*scene = s.clone();
					(curr, next)
				}
			},
			None => {
				let (dispatches, curr, next) = Self::init_inner(frame, s);
				self.persistent = Some(Persistent {
					dispatches,
					scene: s.clone(),
				});
				(curr, next)
			},
		}
	}

	fn init_inner(frame: &mut Frame, s: &RRef<Scene>) -> (PersistentBuffer, Res<BufferHandle>, Res<BufferHandle>) {
		let count = s.meshlet_pointer_count();
		let mut buffer = PersistentBuffer::new(
			frame.device(),
			BufferDesc {
				name: "meshlet dispatch buffer",
				size: (count + 4) as u64 * 2 * std::mem::size_of::<u32>() as u64,
				usage: vk::BufferUsageFlags::INDIRECT_BUFFER
					| vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::TRANSFER_DST,
				on_cpu: false,
			},
		)
		.unwrap();

		let (curr, next) = buffer.next();
		let curr = frame.stage_buffer_new(
			"scene change",
			curr,
			0,
			ByteReader(
				[count, (count + 63) / 64, 1, 1]
					.into_iter()
					.chain(0..count)
					.chain([0, 0, 1, 1])
					.collect(),
			),
		);
		let next = Self::zero_next(frame.pass("clear visbuffer dispatch"), s, next);

		(buffer, curr, next)
	}

	fn zero_next(mut pass: PassBuilder, s: &RRef<Scene>, next: ExternalBuffer) -> Res<BufferHandle> {
		let next = pass.resource(
			next,
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		let count = s.meshlet_pointer_count();
		pass.build(move |mut ctx| unsafe {
			let b = ctx.get(next).buffer;
			ctx.device
				.device()
				.cmd_update_buffer(ctx.buf, b, 0, cast_slice(&[0u32, 0, 1, 1]));
			ctx.device.device().cmd_update_buffer(
				ctx.buf,
				b,
				(4 + count) as u64 * std::mem::size_of::<u32>() as u64,
				cast_slice(&[0u32, 0, 1, 1]),
			);
		});
		next
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let curr_dispatch = pass.get(io.curr_dispatch);
		let next_dispatch = pass.get(io.next_dispatch);
		let mut camera = pass.get(io.camera);
		let visbuffer = pass.get(io.visbuffer);
		let depth = pass.get(io.depth);
		let hzb = io.hzb.map(|d| pass.get(d));

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let mut writer = camera.data.as_mut();
			writer.write(bytes_of(&io.camera_data)).unwrap();

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
						.load_op(if hzb.is_none() {
							vk::AttachmentLoadOp::CLEAR
						} else {
							vk::AttachmentLoadOp::LOAD
						})
						.clear_value(vk::ClearValue {
							color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
						})
						.store_op(vk::AttachmentStoreOp::STORE)])
					.depth_attachment(
						&vk::RenderingAttachmentInfo::default()
							.image_view(depth.view)
							.image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
							.load_op(if hzb.is_none() {
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
					meshlet_pointers: io.meshlet_pointers,
					camera: camera.id.unwrap(),
					meshlet_count: io.meshlet_count,
					width: io.resolution.x,
					height: io.resolution.y,
					curr_dispatch: curr_dispatch.id.unwrap(),
					next_dispatch: next_dispatch.id.unwrap(),
					hzb_sampler: self.hzb_gen.sampler(),
					hzb: hzb.map(|x| x.id.unwrap()),
				}),
			);

			if hzb.is_none() {
				dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.pre_cull_pipeline);
				self.mesh.cmd_draw_mesh_tasks_indirect(
					buf,
					curr_dispatch.buffer,
					1 * std::mem::size_of::<u32>() as u64,
					1,
					std::mem::size_of::<u32>() as u32 * 3,
				);
			} else {
				dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.cull_pipeline);
				self.mesh.cmd_draw_mesh_tasks_indirect(
					buf,
					curr_dispatch.buffer,
					(4 + io.meshlet_count as u64 + 1) * std::mem::size_of::<u32>() as u64,
					1,
					std::mem::size_of::<u32>() as u32 * 3,
				);
			}

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		if let Some(Persistent { dispatches, .. }) = self.persistent {
			dispatches.destroy(device);
		}
		self.hzb_gen.destroy(device);
		device.device().destroy_pipeline(self.pre_cull_pipeline, None);
		device.device().destroy_pipeline(self.cull_pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
