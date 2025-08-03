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

pub struct MeshletCull {
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
	queue: GpuPtr<u8>,
	render: GpuPtr<u8>,
	stats: GpuPtr<CullStats>,
	frame: u64,
	res: Vec2<u32>,
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

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, resources: &Resources) {
		let mut pass = frame.pass("meshlet cull");

		let instances = resources.instances(&mut pass);
		let camera = resources.camera(&mut pass);
		let hzb = resources.hzb(&mut pass);
		let queue = if self.early {
			resources.input_output(&mut pass, resources.meshlet_queue)
		} else {
			resources.input(&mut pass, resources.meshlet_queue)
		};
		let render = resources.output(&mut pass, resources.meshlet_render);
		let stats = resources.stats(&mut pass);

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
				queue: pass.get(queue).ptr(),
				render: pass.get(render).ptr(),
				stats: pass.get(stats).ptr(),
				frame,
				res,
			};
			self.pass.dispatch_indirect(
				&mut pass,
				&push,
				queue,
				std::mem::size_of::<u32>() * if self.early { 2 } else { 6 },
			);
		});
	}

	pub unsafe fn destroy(self) { unsafe { self.pass.destroy(); }}
}
