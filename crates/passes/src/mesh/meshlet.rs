use bytemuck::NoUninit;
use radiance_asset::scene::GpuInstance;
use radiance_graph::{
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		ShaderInfo,
	},
	graph::Frame,
	resource::GpuPtr,
	util::compute::ComputePass,
	Result,
};
use vek::Vec2;

use crate::mesh::{setup::Resources, CameraData, RenderInfo};

pub struct MeshletCull {
	early: bool,
	pass: ComputePass<PushConstants>,
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

impl MeshletCull {
	pub fn new(device: &Device, early: bool) -> Result<Self> {
		Ok(Self {
			early,
			pass: ComputePass::new(
				device,
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
		if self.early {
			resources.output(&mut pass, resources.meshlet_queues[1]);
		}
		let read = resources.input(
			&mut pass,
			if self.early {
				resources.meshlet_queues[0]
			} else {
				resources.meshlet_queues[1]
			},
		);
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

		let instances = info.scene.instances();
		let hzb_sampler = resources.hzb_sampler;
		let late = resources.meshlet_queues[1];
		let res = info.size;
		let len = resources.len;
		pass.build(move |mut pass| {
			let read = pass.get(read);
			self.pass.dispatch_indirect(
				&PushConstants {
					instances,
					camera: pass.get(camera).ptr(),
					hzb: pass.get(hzb).id.unwrap(),
					hzb_sampler,
					read: read.ptr(),
					late: pass.get(late).ptr(),
					hw: pass.get(hw).ptr(),
					sw: pass.get(sw).ptr(),
					res,
					len,
					_pad: 0,
				},
				&pass,
				read.buffer,
				std::mem::size_of::<u32>(),
			);
		});
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
