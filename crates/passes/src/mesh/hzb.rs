use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
		Pipeline,
		ShaderInfo,
	},
	graph::{BufferDesc, BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, PassContext, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView, ImageViewDescUnnamed, ImageViewUsage, Subresource},
	sync::{GlobalBarrier, UsageType},
	Result,
};
use vek::Vec2;

pub struct HzbGen {
	layout: vk::PipelineLayout,
	pipeline: Pipeline,
	pipeline2: Pipeline,
	hzb_sample: vk::Sampler,
	hzb_sample_id: SamplerId,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	visbuffer: StorageImageId,
	outs: [Option<StorageImageId>; 6],
	mips: u32,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants2 {
	mip5: StorageImageId,
	outs: [Option<StorageImageId>; 6],
	mips: u32,
}

struct PassIO {
	visbuffer: Res<ImageView>,
	out: Res<ImageView>,
	size: Vec2<u32>,
	levels: u32,
}

impl HzbGen {
	pub fn new(device: &Device) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::COMPUTE)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?;
			let pipeline = device.compute_pipeline(
				layout,
				ShaderInfo {
					shader: "passes.mesh.hzb.main",
					..Default::default()
				},
			)?;
			let pipeline2 = device.compute_pipeline(
				layout,
				ShaderInfo {
					shader: "passes.mesh.hzb2.main",
					..Default::default()
				},
			)?;

			let hzb_sample = device.device().create_sampler(
				&vk::SamplerCreateInfo::default()
					.min_filter(vk::Filter::LINEAR)
					.mag_filter(vk::Filter::LINEAR)
					.mipmap_mode(vk::SamplerMipmapMode::NEAREST)
					.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
					.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
					.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
					.max_lod(vk::LOD_CLAMP_NONE)
					.push_next(
						&mut vk::SamplerReductionModeCreateInfo::default()
							.reduction_mode(vk::SamplerReductionMode::MIN),
					),
				None,
			)?;
			let hzb_sample_id = device.descriptors().get_sampler(device, hzb_sample);

			Ok(Self {
				layout,
				pipeline,
				pipeline2,
				hzb_sample,
				hzb_sample_id,
			})
		}
	}

	pub fn sampler(&self) -> SamplerId { self.hzb_sample_id }

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, visbuffer: Res<ImageView>, out: Res<ImageView>) {
		let mut pass = frame.pass("generate hzb");
		pass.reference(
			visbuffer,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::ShaderStorageRead(Shader::Compute)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		pass.reference(
			out,
			ImageUsage {
				format: vk::Format::R32_SFLOAT,
				usages: &[
					ImageUsageType::ShaderStorageRead(Shader::Compute),
					ImageUsageType::ShaderStorageWrite(Shader::Compute),
				],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		let desc = pass.desc(visbuffer);
		let size = Vec2::new(desc.size.width, desc.size.height);
		let desc = pass.desc(out);
		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					visbuffer,
					out,
					size,
					levels: desc.levels,
				},
			)
		});
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let visbuffer = pass.get(io.visbuffer);
		let out = pass.get(io.out);

		unsafe {
			let mut outs = [None; 12];
			let mut s = io.size;
			for i in 0..io.levels {
				if s.x > 1 {
					s.x /= 2;
				}
				if s.y > 1 {
					s.y /= 2;
				}
				let dev = pass.device;
				outs[i as usize] = Some(
					pass.get_caches()
						.image_views
						.get(
							dev,
							ImageViewDescUnnamed {
								image: out.image,
								view_type: vk::ImageViewType::TYPE_2D,
								format: vk::Format::R32_SFLOAT,
								usage: ImageViewUsage::Storage,
								size: vk::Extent3D::default().width(s.x).height(s.y).depth(1),
								subresource: Subresource {
									first_mip: i,
									mip_count: 1,
									..Default::default()
								},
							},
						)
						.unwrap()
						.storage_id
						.unwrap(),
				);
			}

			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			let x = (io.size.x + 63) >> 6;
			let y = (io.size.y + 63) >> 6;
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					visbuffer: visbuffer.storage_id.unwrap(),
					outs: [outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]],
					mips: io.levels,
				}),
			);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline.get());
			dev.cmd_dispatch(buf, x, y, 1);
			if io.levels > 6 {
				dev.cmd_pipeline_barrier2(
					buf,
					&vk::DependencyInfo::default().memory_barriers(&[GlobalBarrier {
						previous_usages: &[
							UsageType::ShaderStorageRead(Shader::Compute),
							UsageType::ShaderStorageWrite(Shader::Compute),
						],
						next_usages: &[
							UsageType::ShaderStorageRead(Shader::Compute),
							UsageType::ShaderStorageWrite(Shader::Compute),
						],
					}
					.into()]),
				);
				dev.cmd_push_constants(
					buf,
					self.layout,
					vk::ShaderStageFlags::COMPUTE,
					0,
					bytes_of(&PushConstants2 {
						mip5: outs[5].unwrap(),
						outs: [outs[6], outs[7], outs[8], outs[9], outs[10], outs[11]],
						mips: io.levels,
					}),
				);
				dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline2.get());
				dev.cmd_dispatch(buf, 1, 1, 1);
			}
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			self.pipeline.destroy();
			self.pipeline2.destroy();
			device.device().destroy_pipeline_layout(self.layout, None);
			device.device().destroy_sampler(self.hzb_sample, None);
			device.descriptors().return_sampler(self.hzb_sample_id);
		}
	}
}
