use bytemuck::NoUninit;
use rad_graph::{
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

use crate::{
	mesh::{setup::Resources, CullStats},
	scene::{camera::GpuCamera, virtual_scene::GpuInstance},
};

pub struct InstanceCull {
	early: bool,
	pass: ComputePass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<GpuCamera>,
	hzb: ImageId,
	hzb_sampler: SamplerId,
	next: GpuPtr<u8>,
	late_instances: GpuPtr<u32>,
	stats: GpuPtr<CullStats>,
	frame: u64,
	instance_count: u32,
	res: Vec2<u32>,
	_pad: u32,
}

impl InstanceCull {
	pub fn new(device: &Device, early: bool) -> Result<Self> {
		Ok(Self {
			early,
			pass: ComputePass::new(
				device,
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

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, resources: &Resources) {
		let mut pass = frame.pass("instance cull");

		let instances = resources.instances(&mut pass);
		let camera = resources.camera(&mut pass);
		let hzb = resources.hzb(&mut pass);
		let next = resources.output(&mut pass, resources.bvh_queues[!self.early as usize]);
		let late_instances = if self.early {
			resources.output(&mut pass, resources.late_instances)
		} else {
			resources.input(&mut pass, resources.late_instances)
		};
		let stats = resources.stats(&mut pass);

		let instance_count = resources.scene.instance_count;
		let hzb_sampler = resources.hzb_sampler;
		// TODO: fix
		let frame = 0;
		let res = resources.res;
		pass.build(move |mut pass| {
			let push = PushConstants {
				instances: pass.get(instances).ptr(),
				camera: pass.get(camera).ptr(),
				hzb: pass.get(hzb).id.unwrap(),
				hzb_sampler,
				next: pass.get(next).ptr(),
				late_instances: pass.get(late_instances).ptr(),
				stats: pass.get(stats).ptr(),
				frame,
				instance_count,
				res,
				_pad: 0,
			};
			if self.early {
				self.pass.dispatch(&mut pass, &push, instance_count.div_ceil(64), 1, 1);
			} else {
				self.pass
					.dispatch_indirect(&mut pass, &push, late_instances, std::mem::size_of::<u32>());
			}
		});
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
