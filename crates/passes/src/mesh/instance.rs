use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		Pipeline,
		ShaderInfo,
	},
	graph::{Frame, PassContext, Res},
	resource::{BufferHandle, GpuPtr, ImageView},
	Result,
};
use vek::Vec2;

use crate::{
	asset::scene::GpuInstance,
	mesh::{setup::Resources, CameraData, RenderInfo},
};

pub struct InstanceCull {
	early: bool,
	layout: vk::PipelineLayout,
	pipeline: Pipeline,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	hzb: ImageId,
	hzb_sampler: SamplerId,
	next: GpuPtr<u8>,
	late_instances: GpuPtr<u32>,
	instance_count: u32,
	res: Vec2<u32>,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: GpuPtr<GpuInstance>,
	camera: Res<BufferHandle>,
	hzb: Res<ImageView>,
	hzb_sampler: SamplerId,
	early: Res<BufferHandle>,
	late: Res<BufferHandle>,
	late_instances: Res<BufferHandle>,
	instance_count: u32,
	res: Vec2<u32>,
}

impl InstanceCull {
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
				ShaderInfo {
					shader: "passes.mesh.instance.main",
					spec: if early {
						&["passes.mesh.early"]
					} else {
						&["passes.mesh.late"]
					},
				},
			)?,
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, info: &RenderInfo, resources: &Resources) {
		let mut pass = frame.pass("instance cull");

		let camera = resources.camera(&mut pass);
		let hzb = resources.hzb(&mut pass);
		let early = resources.output(&mut pass, resources.bvh_queues[0]);
		let late = resources.output(&mut pass, resources.bvh_queues[2]);
		let late_instances = if self.early {
			resources.output(&mut pass, resources.late_instances)
		} else {
			resources.input(&mut pass, resources.late_instances)
		};

		let io = PassIO {
			instances: info.scene.instances(),
			camera,
			hzb,
			hzb_sampler: resources.hzb_sampler,
			instance_count: info.scene.instance_count(),
			early,
			late,
			late_instances,
			res: info.size,
		};
		pass.build(move |ctx| self.execute(ctx, io));
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let late_instances = pass.get(io.late_instances);
		unsafe {
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline.get());
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
					camera: pass.get(io.camera).ptr(),
					hzb: pass.get(io.hzb).id.unwrap(),
					hzb_sampler: io.hzb_sampler,
					next: pass.get(if self.early { io.early } else { io.late }).ptr(),
					late_instances: late_instances.ptr(),
					instance_count: io.instance_count,
					res: io.res,
					_pad: 0,
				}),
			);
			if self.early {
				dev.cmd_dispatch(buf, (io.instance_count + 63) / 64, 1, 1);
			} else {
				dev.cmd_dispatch_indirect(buf, late_instances.buffer, std::mem::size_of::<u32>() as _);
			}
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pipeline.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
