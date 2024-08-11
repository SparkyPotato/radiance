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
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		ExternalBuffer,
		ExternalImage,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
		Shader,
	},
	resource::{self, BufferHandle, Image, ImageView, Resource, Subresource},
	util::pipeline::{no_blend, reverse_depth, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::{Mat4, Vec2, Vec4};

use crate::{
	asset::{
		rref::RRef,
		scene::{GpuMeshletPointer, Scene},
	},
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
	early_pipeline: vk::Pipeline,
	late_pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
	mesh: ext::mesh_shader::Device,
}

struct Persistent {
	scene: RRef<Scene>,
	camera: Camera,
	hzb: Image,
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
	meshlet_pointers: BufferId,
	camera: BufferId,
	hzb_sampler: SamplerId,
	hzb: ImageId,
	culled: BufferId,
	meshlet_count: u32,
	size: Vec2<u32>,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: BufferId,
	meshlet_pointers: BufferId,
	camera_data: Camera,
	prev_camera: Camera,
	camera: Res<BufferHandle>,
	hzb: Res<ImageView>,
	culled: Res<BufferHandle>,
	meshlet_count: u32,
	resolution: Vec2<u32>,
	visbuffer: Res<ImageView>,
	depth: Res<ImageView>,
}

impl VisBuffer {
	fn pipeline(device: &Device, layout: vk::PipelineLayout, pre_cull: bool) -> Result<vk::Pipeline> {
		device.graphics_pipeline(&GraphicsPipelineDesc {
			shaders: &[
				device.shader(
					if pre_cull {
						c_str!("radiance-passes/mesh/visbuffer/early")
					} else {
						c_str!("radiance-passes/mesh/visbuffer/late")
					},
					vk::ShaderStageFlags::TASK_EXT,
					None,
				),
				device.shader(
					if pre_cull {
						c_str!("radiance-passes/mesh/visbuffer/earlym")
					} else {
						c_str!("radiance-passes/mesh/visbuffer/latem")
					},
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
				early_pipeline: Self::pipeline(device, layout, true)?,
				late_pipeline: Self::pipeline(device, layout, false)?,
				layout,
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let (hzb, culled, prev_camera) = self.init_persistent(frame, &info);
		self.persistent.as_mut().unwrap().camera = info.camera;

		let mut pass = frame.pass("visbuffer early");
		let usage = ImageUsage {
			format: vk::Format::R32_SFLOAT,
			usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Task)],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};
		let hzb = match hzb {
			Ok(hzb) => {
				pass.reference(hzb, usage);
				hzb
			},
			Err(hzb) => pass.resource(hzb, usage),
		};
		let c = pass.resource(
			BufferDesc {
				size: std::mem::size_of::<CameraData>() as u64 * 2,
				upload: true,
			},
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::ShaderStorageRead(Shader::Mesh),
				],
			},
		);
		pass.reference(
			culled,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::ShaderStorageWrite(Shader::Task),
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

		let io = PassIO {
			instances: info.scene.instances(),
			meshlet_pointers: info.scene.meshlet_pointers(),
			camera_data: info.camera,
			prev_camera,
			hzb,
			culled,
			meshlet_count: info.scene.meshlet_pointer_count(),
			resolution: info.size,
			camera: c,
			visbuffer,
			depth,
		};
		let this: &Self = self;
		pass.build(move |ctx| this.execute(ctx, io, true));

		this.hzb_gen.run(frame, depth, hzb);

		let mut pass = frame.pass("visbuffer late");
		pass.reference(
			c,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Task),
					BufferUsageType::ShaderStorageRead(Shader::Mesh),
				],
			},
		);
		pass.reference(
			culled,
			BufferUsage {
				usages: &[
					BufferUsageType::IndirectBuffer,
					BufferUsageType::ShaderStorageRead(Shader::Task),
				],
			},
		);
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
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Task)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		pass.build(move |ctx| this.execute(ctx, io, false));

		this.hzb_gen.run(frame, depth, hzb);

		visbuffer
	}

	fn init_persistent(
		&mut self, frame: &mut Frame, info: &RenderInfo,
	) -> (
		std::result::Result<Res<ImageView>, ExternalImage>,
		Res<BufferHandle>,
		Camera,
	) {
		let needs_clear = match &mut self.persistent {
			Some(Persistent { scene, hzb, .. }) => {
				if hzb.desc().size.width != info.size.x / 2 || hzb.desc().size.height != info.size.y / 2 {
					frame.delete(self.persistent.take().unwrap().hzb);
					self.persistent = Some(Persistent {
						scene: info.scene.clone(),
						camera: info.camera,
						hzb: Self::make_hzb(frame.device(), info.size / 2),
					});
					true
				} else {
					let r = !scene.ptr_eq(&info.scene);
					if r {
						*scene = info.scene.clone();
					}
					r
				}
			},
			None => {
				self.persistent = Some(Persistent {
					scene: info.scene.clone(),
					camera: info.camera,
					hzb: Self::make_hzb(frame.device(), info.size),
				});
				true
			},
		};

		let Persistent { camera, hzb, .. } = self.persistent.as_ref().unwrap();
		let mut pass = frame.pass("setup cull");
		let hzb = if needs_clear {
			Ok(pass.resource(
				ExternalImage {
					handle: hzb.handle(),
					layout: vk::ImageLayout::UNDEFINED,
					desc: hzb.desc(),
				},
				ImageUsage {
					format: vk::Format::UNDEFINED,
					usages: &[ImageUsageType::TransferWrite],
					view_type: None,
					subresource: Subresource::default(),
				},
			))
		} else {
			Err(ExternalImage {
				handle: hzb.handle(),
				layout: vk::ImageLayout::GENERAL,
				desc: hzb.desc(),
			})
		};
		let culled = pass.resource(
			BufferDesc {
				size: (info.scene.meshlet_pointer_count() + 4) as u64 * std::mem::size_of::<u32>() as u64,
				upload: false,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |mut ctx| unsafe {
			let dev = ctx.device.device();
			let buf = ctx.buf;
			let culled = ctx.get(culled);

			if let Ok(hzb) = hzb {
				let hzb = ctx.get(hzb);
				dev.cmd_clear_color_image(
					buf,
					hzb.image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&vk::ClearColorValue::default(),
					&[vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.base_array_layer(0)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)],
				);
			}

			dev.cmd_update_buffer(buf, culled.buffer, 0, bytes_of(&[0u32, 0, 1, 1]));
		});

		(hzb, culled, *camera)
	}

	fn make_hzb(device: &Device, size: Vec2<u32>) -> Image {
		Image::create(
			device,
			resource::ImageDesc {
				name: "persistent hzb",
				flags: vk::ImageCreateFlags::empty(),
				format: vk::Format::R32_SFLOAT,
				size: vk::Extent3D {
					width: size.x,
					height: size.y,
					depth: 1,
				},
				levels: size.x.max(size.y).ilog2(),
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
			},
		)
		.unwrap()
	}

	fn execute(&self, mut pass: PassContext, io: PassIO, is_early: bool) {
		let mut camera = pass.get(io.camera);
		let visbuffer = pass.get(io.visbuffer);
		let depth = pass.get(io.depth);
		let hzb = pass.get(io.hzb);
		let culled = pass.get(io.culled);

		let dev = pass.device.device();
		let buf = pass.buf;

		unsafe {
			if is_early {
				let mut writer = camera.data.as_mut();
				let aspect = io.resolution.x as f32 / io.resolution.y as f32;
				let cd = CameraData::new(aspect, io.camera_data);
				let prev_cd = CameraData::new(aspect, io.prev_camera);
				writer.write(bytes_of(&cd)).unwrap();
				writer.write(bytes_of(&prev_cd)).unwrap();
			}

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
					meshlet_pointers: io.meshlet_pointers,
					camera: camera.id.unwrap(),
					hzb_sampler: self.hzb_gen.sampler(),
					hzb: hzb.id.unwrap(),
					culled: culled.id.unwrap(),
					meshlet_count: io.meshlet_count,
					size: io.resolution,
				}),
			);

			if is_early {
				dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.early_pipeline);
				self.mesh.cmd_draw_mesh_tasks(buf, (io.meshlet_count + 63) / 64, 1, 1);
			} else {
				dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::GRAPHICS, self.late_pipeline);
				self.mesh.cmd_draw_mesh_tasks_indirect(
					buf,
					culled.buffer,
					std::mem::size_of::<u32>() as u64,
					1,
					std::mem::size_of::<u32>() as u32 * 3,
				);
			}

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		if let Some(Persistent { hzb, .. }) = self.persistent {
			hzb.destroy(device);
		}
		self.hzb_gen.destroy(device);
		device.device().destroy_pipeline(self.early_pipeline, None);
		device.device().destroy_pipeline(self.late_pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
