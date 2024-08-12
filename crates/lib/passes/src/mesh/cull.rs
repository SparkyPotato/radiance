use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
	},
	graph::{
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		ExternalImage,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Res,
	},
	resource::{BufferHandle, Image, ImageView, Resource, Subresource},
	sync::Shader,
	Result,
};
use radiance_shader_compiler::c_str;
use vek::Vec2;

use crate::{
	asset::{rref::RRef, scene::Scene},
	mesh::{Camera, CameraData},
};

pub struct Cull {
	early: bool,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	camera: BufferId,
	hzb_sampler: SamplerId,
	hzb: ImageId,
	culled: BufferId,
	o: BufferId,
	meshlet_count: u32,
	instance_count: u32,
	size: Vec2<u32>,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: BufferId,
	camera_data: Camera,
	prev_camera: Camera,
	camera: Res<BufferHandle>,
	hzb: Res<ImageView>,
	hzb_sampler: SamplerId,
	culled: Res<BufferHandle>,
	o: Res<BufferHandle>,
	meshlet_count: u32,
	instance_count: u32,
	resolution: Vec2<u32>,
}

#[derive(Copy, Clone)]
pub struct CullInfo<'a> {
	pub hzb: std::result::Result<Res<ImageView>, &'a Image>,
	pub hzb_sampler: SamplerId,
	pub needs_clear: bool,
	pub scene: &'a RRef<Scene>,
	pub culled: Option<Res<BufferHandle>>,
	pub cam_buf: Option<Res<BufferHandle>>,
	pub camera: Camera,
	pub prev_camera: Camera,
	pub resolution: Vec2<u32>,
}

impl Cull {
	pub fn new(device: &Device, early: bool) -> Result<Self> {
		let layout = unsafe {
			device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::COMPUTE)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?
		};
		Ok(Self {
			early,
			layout,
			pipeline: device.compute_pipeline(
				layout,
				device.shader(
					if early {
						c_str!("radiance-passes/mesh/visbuffer/early")
					} else {
						c_str!("radiance-passes/mesh/visbuffer/late")
					},
					vk::ShaderStageFlags::COMPUTE,
					None,
				),
			)?,
		})
	}

	fn setup_cull(
		frame: &mut Frame, early: bool, hzb: std::result::Result<Res<ImageView>, &Image>, needs_clear: bool,
		scene: &RRef<Scene>,
	) -> (
		Res<BufferHandle>,
		Option<Res<BufferHandle>>,
		std::result::Result<Res<ImageView>, ExternalImage>,
	) {
		let mut pass = frame.pass(if early { "setup early cull" } else { "setup late cull" });

		let o = pass.resource(
			BufferDesc {
				size: (scene.meshlet_count() + 3) as u64 * std::mem::size_of::<u32>() as u64,
				upload: false,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);

		let culled = early.then(|| {
			pass.resource(
				BufferDesc {
					size: (scene.meshlet_count() + 4) as u64 * std::mem::size_of::<u32>() as u64,
					upload: false,
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});

		let hzb = match hzb {
			Ok(hzb) => Ok(hzb),
			Err(hzb) => {
				if needs_clear {
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
				}
			},
		};

		pass.build(move |mut ctx| unsafe {
			ctx.device
				.device()
				.cmd_update_buffer(ctx.buf, ctx.get(o).buffer, 0, bytes_of(&[0u32, 1, 1]));
			culled.map(|c| {
				ctx.device
					.device()
					.cmd_update_buffer(ctx.buf, ctx.get(c).buffer, 0, bytes_of(&[0u32, 0, 1, 1]))
			});
			if let Ok(hzb) = hzb {
				if !needs_clear {
					return;
				}
				ctx.device.device().cmd_clear_color_image(
					ctx.buf,
					ctx.get(hzb).image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&vk::ClearColorValue::default(),
					&[vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.base_array_layer(0)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)],
				)
			}
		});

		(o, culled, hzb)
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, info: CullInfo,
	) -> (Res<BufferHandle>, Res<BufferHandle>, Res<BufferHandle>, Res<ImageView>) {
		let (o, c, hzb) = Self::setup_cull(frame, self.early, info.hzb, info.needs_clear, info.scene);
		let culled = c.unwrap_or_else(|| info.culled.unwrap());

		let mut pass = frame.pass(if self.early { "early cull" } else { "late cull" });

		let usage = BufferUsage {
			usages: &[BufferUsageType::ShaderStorageRead(Shader::Compute)],
		};
		let c = match info.cam_buf {
			Some(c) => {
				pass.reference(c, usage);
				c
			},
			None => pass.resource(
				BufferDesc {
					size: std::mem::size_of::<CameraData>() as u64 * 2,
					upload: true,
				},
				usage,
			),
		};

		let usage = ImageUsage {
			format: vk::Format::R32_SFLOAT,
			usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Compute)],
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
		pass.reference(
			o,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);
		pass.reference(
			culled,
			BufferUsage {
				usages: if self.early {
					&[
						BufferUsageType::ShaderStorageRead(Shader::Compute),
						BufferUsageType::ShaderStorageWrite(Shader::Compute),
					]
				} else {
					&[
						BufferUsageType::IndirectBuffer,
						BufferUsageType::ShaderStorageRead(Shader::Compute),
					]
				},
			},
		);

		let io = PassIO {
			instances: info.scene.instances(),
			camera_data: info.camera,
			prev_camera: info.prev_camera,
			camera: c,
			hzb,
			hzb_sampler: info.hzb_sampler,
			culled,
			o,
			meshlet_count: info.scene.meshlet_count(),
			instance_count: info.scene.instance_count(),
			resolution: info.resolution,
		};
		pass.build(move |ctx| self.execute(ctx, io));

		(o, culled, c, hzb)
	}

	fn execute(&self, mut ctx: PassContext, io: PassIO) {
		let dev = ctx.device.device();
		let buf = ctx.buf;
		let culled = ctx.get(io.culled);
		let mut camera = ctx.get(io.camera);
		unsafe {
			if self.early {
				let mut writer = camera.data.as_mut();
				let aspect = io.resolution.x as f32 / io.resolution.y as f32;
				let cd = CameraData::new(aspect, io.camera_data);
				let prev_cd = CameraData::new(aspect, io.prev_camera);
				writer.write(bytes_of(&cd)).unwrap();
				writer.write(bytes_of(&prev_cd)).unwrap();
			}

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[ctx.device.descriptors().set()],
				&[],
			);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					instances: io.instances,
					camera: camera.id.unwrap(),
					hzb_sampler: io.hzb_sampler,
					hzb: ctx.get(io.hzb).id.unwrap(),
					culled: culled.id.unwrap(),
					o: ctx.get(io.o).id.unwrap(),
					meshlet_count: io.meshlet_count,
					instance_count: io.instance_count,
					size: io.resolution,
				}),
			);
			if self.early {
				dev.cmd_dispatch(buf, (io.meshlet_count + 63) / 64, 1, 1);
			} else {
				dev.cmd_dispatch_indirect(buf, culled.buffer, std::mem::size_of::<u32>() as _);
			}
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
