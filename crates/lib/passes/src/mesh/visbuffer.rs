use std::io::Write;

use ash::{extensions::ext, vk};
use bytemuck::{bytes_of, cast_slice, NoUninit};
use radiance_asset_runtime::{rref::RRef, scene::Scene};
use radiance_core::{
	persistent::PersistentBuffer,
	pipeline::GraphicsPipelineDesc,
	CoreDevice,
	CoreFrame,
	CorePass,
	PassBuilderExt,
	RenderCore,
};
use radiance_graph::{
	device::descriptor::BufferId,
	graph::{BufferUsage, BufferUsageType, ImageDesc, ImageUsage, ImageUsageType, Res, Shader, UploadBufferDesc},
	resource::{GpuBufferHandle, ImageView, UploadBufferHandle},
	Result,
};
use radiance_shader_compiler::c_str;
use radiance_util::pipeline::{no_blend, reverse_depth, simple_blend};
use vek::{Mat4, Ray, Vec2, Vec4};

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
	vis_pipeline: vk::Pipeline,
	invis_pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	mesh: ext::MeshShader,
	workgroups: PersistentBuffer,
	visibility: Option<PersistentBuffer>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct CameraData {
	view: Mat4<f32>,
	proj: Mat4<f32>,
	view_proj: Mat4<f32>,
}

impl CameraData {
	fn new(aspect: f32, camera: Camera) -> Self {
		let proj = infinite_projection(aspect, camera.fov, camera.near);
		let view = camera.view;
		let view_proj = proj * view;

		Self { view, proj, view_proj }
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	meshlet_pointers: BufferId,
	rw: BufferId,
	ww: BufferId,
	rd: BufferId,
	wd: BufferId,
	camera: BufferId,
	meshlet_count: u32,
}

struct PassIO {
	instances: BufferId,
	meshlet_pointers: BufferId,
	rw: Res<GpuBufferHandle>,
	ww: Res<GpuBufferHandle>,
	rd: Res<GpuBufferHandle>,
	wd: Res<GpuBufferHandle>,
	cull_camera: CameraData,
	draw_camera: CameraData,
	meshlet_count: u32,
	camera: Res<UploadBufferHandle>,
	visbuffer: Res<ImageView>,
	depth: Res<ImageView>,
}

impl VisBuffer {
	fn pipeline(device: &CoreDevice, core: &RenderCore, layout: vk::PipelineLayout, vis: bool) -> Result<vk::Pipeline> {
		core.graphics_pipeline(
			device,
			&GraphicsPipelineDesc {
				shaders: &[
					core.shaders
						.shader(
							if vis {
								c_str!("radiance-passes/mesh/visbuffer/visible")
							} else {
								c_str!("radiance-passes/mesh/visbuffer/invisible")
							},
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
		)
	}

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

			let vis_pipeline = Self::pipeline(device, core, layout, true)?;
			let invis_pipeline = Self::pipeline(device, core, layout, false)?;

			Ok(Self {
				vis_pipeline,
				invis_pipeline,
				layout,
				mesh: ext::MeshShader::new(device.instance(), device.device()),
				workgroups: PersistentBuffer::new(
					device,
					32,
					vk::BufferUsageFlags::INDIRECT_BUFFER
						| vk::BufferUsageFlags::TRANSFER_DST
						| vk::BufferUsageFlags::STORAGE_BUFFER,
				)?,
				visibility: None,
			})
		}
	}

	pub fn init_visibility<'pass>(
		&mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, info: RenderInfo,
	) -> (Res<GpuBufferHandle>, Res<GpuBufferHandle>) {
		let new = self.visibility(device, &info.scene);
		let work = &mut self.workgroups;
		let data = self.visibility.as_mut().unwrap();
		let mut pass = frame.pass("init visibility");

		let mut usages: &[_] = &[];
		if new {
			let values: Vec<_> = (0..info.scene.meshlet_pointer_count()).collect();
			pass.stage(device, |cx, _| {
				cx.stage_buffer(cast_slice(&values), data.buffers[data.current].inner(), 0)
			})
			.unwrap();
			usages = &[BufferUsageType::TransferWrite];
		}
		let (rw, ww) = work.next(
			&mut pass,
			BufferUsage { usages },
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);

		pass.build(move |mut ctx| unsafe {
			let rw = ctx.get(rw);
			let ww = ctx.get(ww);
			let dev = ctx.device.device();
			let buf = ctx.buf;

			if new {
				dev.cmd_update_buffer(
					buf,
					rw.buffer,
					0,
					bytes_of(&[
						info.scene.meshlet_pointer_count(),
						(info.scene.meshlet_pointer_count() + 63) / 64,
						1,
						1,
						0,
						0,
						1,
						1,
					]),
				);
			}
			dev.cmd_update_buffer(buf, ww.buffer, 0, &cast_slice(&[0u32, 0, 1, 1, 0, 0, 1, 1]));
		});

		(rw, ww)
	}

	pub fn run<'pass>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, info: RenderInfo,
	) -> Res<ImageView> {
		let (rw, ww) = self.init_visibility(device, frame, info.clone());

		let mut pass = frame.pass("visbuffer");
		pass.input(
			rw,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::IndirectBuffer,
				],
			},
		);
		pass.input(
			ww,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::ShaderStorageWrite(Shader::Task),
				],
			},
		);
		let data = self.visibility.as_mut().unwrap();
		let (rd, wd) = data.next(
			&mut pass,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Task)],
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageWrite(Shader::Task)],
			},
		);

		let aspect = info.size.x as f32 / info.size.y as f32;
		let draw_camera = CameraData::new(aspect, info.camera);
		let cull_camera = info
			.cull_camera
			.map(|c| CameraData::new(aspect, c))
			.unwrap_or(draw_camera);

		let origin = Vec4::new(0.0, 0.0, 0.0, 1.0);
		let dir = Vec4::new(0.0, 0.0, 1.0, 0.0);
		let mat = draw_camera.view.inverted();
		let ray = Ray {
			origin: (mat * origin).xyz(),
			direction: (mat * dir).xyz().normalized(),
		};

		let c = pass.output(
			UploadBufferDesc {
				size: (std::mem::size_of::<CameraData>() * 2) as _,
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
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
		};
		let visbuffer = pass.output(
			desc,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);
		let depth = pass.output(
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
					instances: info.scene.instances(),
					meshlet_pointers: info.scene.meshlet_pointers(),
					rw,
					ww,
					rd,
					wd,
					cull_camera,
					draw_camera,
					meshlet_count: info.scene.meshlet_pointer_count(),
					camera: c,
					visbuffer,
					depth,
				},
			)
		});

		visbuffer
	}

	fn execute(&self, mut pass: CorePass, io: PassIO) {
		let mut camera = pass.get(io.camera);
		let visbuffer = pass.get(io.visbuffer);
		let depth = pass.get(io.depth);
		let rw = pass.get(io.rw);
		let ww = pass.get(io.ww);
		let rd = pass.get(io.rd);
		let wd = pass.get(io.wd);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			let mut writer = camera.data.as_mut();
			writer.write(bytes_of(&io.cull_camera)).unwrap();
			writer.write(bytes_of(&io.draw_camera)).unwrap();

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
					rw: rw.id.unwrap(),
					ww: ww.id.unwrap(),
					rd: rd.id.unwrap(),
					wd: wd.id.unwrap(),
					camera: camera.id.unwrap(),
					meshlet_count: io.meshlet_count,
				}),
			);

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.vis_pipeline);
			self.mesh.cmd_draw_mesh_tasks_indirect(buf, rw.buffer, 4, 1, 12);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.invis_pipeline);
			self.mesh.cmd_draw_mesh_tasks_indirect(buf, rw.buffer, 20, 1, 12);

			dev.cmd_end_rendering(buf);
		}
	}

	fn visibility(&mut self, device: &CoreDevice, scene: &Scene) -> bool {
		let mut new = false;
		let size = scene.meshlet_pointer_count() as u64 * 2 * 4;
		if self.visibility.is_none() || self.visibility.as_ref().unwrap().size() < size {
			new = true;
			if let Some(old) = self.visibility.replace(
				PersistentBuffer::new(
					device,
					size,
					vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				)
				.unwrap(),
			) {
				unsafe { old.destroy(device) }
			}
		}
		new
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		device.device().destroy_pipeline(self.vis_pipeline, None);
		device.device().destroy_pipeline(self.invis_pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
		self.workgroups.destroy(device);
		if let Some(visibility) = self.visibility {
			visibility.destroy(device);
		}
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
