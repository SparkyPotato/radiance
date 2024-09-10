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

pub struct BvhCull {
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
	next: GpuPtr<u8>,
	meshlet: GpuPtr<u8>,
	late: GpuPtr<u8>,
	late_meshlet: GpuPtr<u8>,
	res: Vec2<u32>,
	len: u32,
	_pad: u32,
}

impl BvhCull {
	pub fn new(device: &Device, early: bool) -> Result<Self> {
		Ok(Self {
			early,
			pass: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.mesh.bvh.main",
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
		let mut read = if self.early {
			resources.bvh_queues[0]
		} else {
			resources.bvh_queues[2]
		};
		let mut next = resources.bvh_queues[1];
		for _ in 0..info.scene.max_depth() {
			let mut pass = frame.pass("bvh cull");

			let camera = resources.camera(&mut pass);
			let hzb = resources.hzb(&mut pass);
			if self.early {
				resources.output(&mut pass, resources.bvh_queues[2]);
			}
			resources.input(&mut pass, read);
			resources.output(&mut pass, next);
			let q = if self.early {
				resources.output(&mut pass, resources.meshlet_queues[1]);
				resources.meshlet_queues[0]
			} else {
				resources.meshlet_queues[1]
			};
			let meshlet = resources.output(&mut pass, q);

			let instances = info.scene.instances();
			let hzb_sampler = resources.hzb_sampler;
			let late = resources.bvh_queues[2];
			let late_meshlet = resources.meshlet_queues[1];
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
						next: pass.get(next).ptr(),
						meshlet: pass.get(meshlet).ptr(),
						late: pass.get(late).ptr(),
						late_meshlet: pass.get(late_meshlet).ptr(),
						res,
						len,
						_pad: 0,
					},
					&pass,
					read.buffer,
					std::mem::size_of::<u32>(),
				);
			});

			(read, next) = (next, read);
		}
	}

	pub unsafe fn destroy(self, device: &Device) { self.pass.destroy(device); }
}
