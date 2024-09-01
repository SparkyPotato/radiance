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

pub struct MeshletCull {
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
	read: GpuPtr<u8>,
	late: GpuPtr<u8>,
	hw: GpuPtr<u8>,
	sw: GpuPtr<u8>,
	res: Vec2<u32>,
	len: u32,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	instances: GpuPtr<GpuInstance>,
	camera: Res<BufferHandle>,
	hzb: Res<ImageView>,
	hzb_sampler: SamplerId,
	len: u32,
	read: Res<BufferHandle>,
	late: Res<BufferHandle>,
	hw: Res<BufferHandle>,
	sw: Res<BufferHandle>,
	res: Vec2<u32>,
}

impl MeshletCull {
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
					shader: "passes.mesh.meshlet.main",
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
		let mut pass = frame.pass("meshlet cull");

		let camera = resources.camera(&mut pass);
		let hzb = resources.hzb(&mut pass);
		let read = resources.input(
			&mut pass,
			if self.early {
				resources.meshlet_queues[0]
			} else {
				resources.meshlet_queues[1]
			},
		);
		let late = resources.output(&mut pass, resources.meshlet_queues[1]);
		let hw = resources.output(
			&mut pass,
			if self.early {
				resources.meshlet_render_lists[0]
			} else {
				resources.meshlet_render_lists[2]
			},
		);
		let sw = resources.output(
			&mut pass,
			if self.early {
				resources.meshlet_render_lists[1]
			} else {
				resources.meshlet_render_lists[3]
			},
		);

		let io = PassIO {
			instances: info.scene.instances(),
			camera,
			hzb,
			hzb_sampler: resources.hzb_sampler,
			len: resources.len,
			read,
			late,
			hw,
			sw,
			res: info.size,
		};
		pass.build(move |ctx| self.execute(ctx, io));
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let read = pass.get(io.read);
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
					read: read.ptr(),
					late: pass.get(io.late).ptr(),
					hw: pass.get(io.hw).ptr(),
					sw: pass.get(io.sw).ptr(),
					res: io.res,
					len: io.len,
					_pad: 0,
				}),
			);
			dev.cmd_dispatch_indirect(buf, read.buffer, std::mem::size_of::<u32>() as _);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pipeline.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
