use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
	},
	graph::{BufferUsage, BufferUsageType, Frame, PassContext, Res},
	resource::{BufferHandle, ImageView},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::Vec2;

use crate::mesh::{setup::Resources, RenderInfo};

pub struct BvhCull {
	early: bool,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: BufferId,
	camera: BufferId,
	hzb: ImageId,
	hzb_sampler: SamplerId,
	read: BufferId,
	next: BufferId,
	meshlet: BufferId,
	late: BufferId,
	late_meshlet: BufferId,
	res: Vec2<u32>,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: BufferId,
	camera: Res<BufferHandle>,
	hzb: Res<ImageView>,
	hzb_sampler: SamplerId,
	read: Res<BufferHandle>,
	next: Res<BufferHandle>,
	meshlet: Res<BufferHandle>,
	late: Res<BufferHandle>,
	late_meshlet: Res<BufferHandle>,
	res: Vec2<u32>,
}

impl BvhCull {
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
						c_str!("radiance-passes/mesh/bvh_early")
					} else {
						c_str!("radiance-passes/mesh/bvh_late")
					},
					vk::ShaderStageFlags::COMPUTE,
					None,
				),
			)?,
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, info: &RenderInfo, resources: &Resources) {
		let mut read = if self.early {
			resources.bvh_queues[0]
		} else {
			resources.bvh_queues[2]
		};
		let mut next = resources.bvh_queues[1];
		for i in 0..info.scene.max_depth() {
			let mut pass = frame.pass("bvh cull");

			let camera = resources.camera(&mut pass);
			let hzb = resources.hzb(&mut pass);
			let late = resources.output(&mut pass, resources.bvh_queues[2]);
			resources.input(&mut pass, read);
			resources.output(&mut pass, next);
			let meshlet = resources.output(&mut pass, resources.meshlet_queues[0]);
			let late_meshlet = resources.output(&mut pass, resources.meshlet_queues[1]);

			let io = PassIO {
				instances: info.scene.instances(),
				camera,
				hzb,
				hzb_sampler: resources.hzb_sampler,
				read,
				next,
				meshlet,
				late,
				late_meshlet,
				res: info.size,
			};
			pass.build(move |pass| self.execute(pass, io));

			if i != info.scene.max_depth() - 1 {
				let mut pass = frame.pass("clear buf");
				pass.reference(
					read,
					BufferUsage {
						usages: &[BufferUsageType::TransferWrite],
					},
				);
				pass.build(move |mut pass| unsafe {
					pass.device.device().cmd_update_buffer(
						pass.buf,
						pass.get(read).buffer,
						0,
						bytes_of(&[0u32, 0, 1, 1]),
					);
				})
			}

			(read, next) = (next, read);
		}
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let read = pass.get(io.read);
		unsafe {
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					instances: io.instances,
					camera: pass.get(io.camera).id.unwrap(),
					hzb: pass.get(io.hzb).id.unwrap(),
					hzb_sampler: io.hzb_sampler,
					read: read.id.unwrap(),
					next: pass.get(io.next).id.unwrap(),
					meshlet: pass.get(io.meshlet).id.unwrap(),
					late: pass.get(io.late).id.unwrap(),
					late_meshlet: pass.get(io.late_meshlet).id.unwrap(),
					res: io.res,
				}),
			);
			dev.cmd_dispatch_indirect(buf, read.buffer, std::mem::size_of::<u32>() as _);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
