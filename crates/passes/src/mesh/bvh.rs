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

use crate::mesh::{setup::Resources, CameraData};

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
	queue: GpuPtr<u8>,
	late: GpuPtr<u8>,
	meshlet: GpuPtr<u8>,
	frame: u64,
	res: Vec2<u32>,
	ping: u32,
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

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, resources: &Resources) {
		let queue = resources.bvh_queues[!self.early as usize];
		let late = resources.bvh_queues[1];
		let mut ping = true;

		for _ in 0..resources.scene.max_depth {
			let mut pass = frame.pass("bvh cull");

			let instances = resources.instances(&mut pass);
			let camera = resources.camera(&mut pass);
			let hzb = resources.hzb(&mut pass);
			if self.early {
				resources.output(&mut pass, late);
			}
			resources.input_output(&mut pass, queue);
			let meshlet = resources.output(&mut pass, resources.meshlet_queue);

			let hzb_sampler = resources.hzb_sampler;
			let frame = resources.scene.frame;
			let res = resources.res;
			pass.build(move |mut pass| {
				let queue = pass.get(queue);
				self.pass.dispatch_indirect(
					&PushConstants {
						instances: pass.get(instances).ptr(),
						camera: pass.get(camera).ptr(),
						hzb: pass.get(hzb).id.unwrap(),
						hzb_sampler,
						queue: queue.ptr(),
						meshlet: pass.get(meshlet).ptr(),
						late: pass.get(late).ptr(),
						frame,
						res,
						ping: ping as _,
						_pad: 0,
					},
					&pass,
					queue.buffer,
					std::mem::size_of::<u32>() * if ping { 2 } else { 6 },
				);
			});

			ping = !ping;
		}
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
