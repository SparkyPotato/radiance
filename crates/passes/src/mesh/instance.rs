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

use crate::mesh::{setup::Resources, CameraData, CullStats, RenderInfo};

pub struct InstanceCull {
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
	next: GpuPtr<u8>,
	late_instances: GpuPtr<u32>,
	stats: GpuPtr<CullStats>,
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

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, info: &RenderInfo, resources: &Resources) {
		let mut pass = frame.pass("instance cull");

		let camera = resources.camera(&mut pass);
		let hzb = resources.hzb(&mut pass);
		let next = resources.output(&mut pass, resources.bvh_queues[!self.early as usize]);
		let late_instances = if self.early {
			resources.output(&mut pass, resources.late_instances)
		} else {
			resources.input(&mut pass, resources.late_instances)
		};
		let stats = resources.stats(&mut pass);

		let instances = info.scene.instances();
		let hzb_sampler = resources.hzb_sampler;
		let instance_count = info.scene.instance_count();
		let res = info.size;
		pass.build(move |mut pass| {
			let latei = pass.get(late_instances);
			let push = PushConstants {
				instances,
				camera: pass.get(camera).ptr(),
				hzb: pass.get(hzb).id.unwrap(),
				hzb_sampler,
				next: pass.get(next).ptr(),
				late_instances: latei.ptr(),
				stats: pass.get(stats).ptr(),
				instance_count,
				res,
				_pad: 0,
			};
			if self.early {
				self.pass.dispatch(&push, &pass, (instance_count + 63) / 64, 1, 1);
			} else {
				self.pass
					.dispatch_indirect(&push, &pass, latei.buffer, std::mem::size_of::<u32>());
			}
		});
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
