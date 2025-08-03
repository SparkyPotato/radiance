use ash::{ext, vk};
use bytemuck::{NoUninit, Pod, Zeroable};
use rad_graph::{
	device::{descriptor::StorageImageId, Device, GraphicsPipelineDesc, ShaderInfo},
	graph::{BufferUsage, Frame, ImageUsage, PassBuilder, PassContext, Res},
	resource::{BufferHandle, GpuPtr, ImageView},
	sync::Shader,
	util::{compute::ComputePass, render::RenderPass},
	Result,
};
use vek::Vec2;

pub use crate::mesh::setup::{DebugRes, DebugResId};
use crate::{
	mesh::{bvh::BvhCull, hzb::HzbGen, instance::InstanceCull, meshlet::MeshletCull, setup::Setup},
	scene::{camera::GpuCamera, virtual_scene::GpuInstance, WorldRenderer},
};

mod bvh;
mod hzb;
mod instance;
mod meshlet;
mod setup;

#[derive(Clone)]
pub struct RenderInfo {
	pub size: Vec2<u32>,
	pub debug_info: bool,
}

#[derive(Copy, Clone)]
pub struct VisBufferReader {
	pub visbuffer: Res<ImageView>,
	pub queue: Res<BufferHandle>,
	pub debug: Option<DebugRes>,
}

impl VisBufferReader {
	pub fn add(&self, pass: &mut PassBuilder, shader: Shader, debug: bool) {
		pass.reference(self.queue, BufferUsage::read(shader));

		let usage = ImageUsage::read_2d(shader);
		pass.reference(self.visbuffer, usage);
		if let Some(d) = self.debug
			&& debug
		{
			pass.reference(d.overdraw, usage);
			pass.reference(d.hwsw, usage);
		}
	}

	pub fn get(self, pass: &mut PassContext) -> GpuVisBufferReader {
		GpuVisBufferReader {
			queue: pass.get(self.queue).ptr(),
			visbuffer: pass.get(self.visbuffer).storage_id.unwrap(),
			_pad: 0,
		}
	}

	pub fn get_debug(self, pass: &mut PassContext) -> GpuVisBufferReaderDebug {
		GpuVisBufferReaderDebug {
			queue: pass.get(self.queue).ptr(),
			visbuffer: pass.get(self.visbuffer).storage_id.unwrap(),
			debug: self.debug.map(|x| x.get(pass)),
			_pad: 0,
		}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuVisBufferReaderDebug {
	queue: GpuPtr<u8>,
	visbuffer: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuVisBufferReader {
	queue: GpuPtr<u8>,
	visbuffer: StorageImageId,
	_pad: u32,
}

#[derive(Copy, Clone)]
pub struct RenderOutput {
	pub stats: CullStats,
	pub instances: Res<BufferHandle>,
	pub camera: Res<BufferHandle>,
	pub reader: VisBufferReader,
}

pub struct VisBuffer {
	setup: Setup,
	early_instance_cull: InstanceCull,
	late_instance_cull: InstanceCull,
	early_bvh_cull: BvhCull,
	late_bvh_cull: BvhCull,
	early_meshlet_cull: MeshletCull,
	late_meshlet_cull: MeshletCull,
	hzb_gen: HzbGen,
	no_debug: Passes,
	debug: Passes,
	mesh: ext::mesh_shader::Device,
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct PassStats {
	pub instances: u32,
	pub candidate_meshlets: u32,
	pub hw_meshlets: u32,
	pub sw_meshlets: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct CullStats {
	pub early: PassStats,
	pub late: PassStats,
	pub overflow: u32,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<GpuCamera>,
	queue: GpuPtr<u8>,
	stats: GpuPtr<CullStats>,
	output: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	early: bool,
	instances: Res<BufferHandle>,
	queue: Res<BufferHandle>,
	camera: Res<BufferHandle>,
	stats: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	debug: Option<DebugRes>,
}

struct Passes {
	early_hw: RenderPass<PushConstants>,
	early_sw: ComputePass<PushConstants>,
	late_hw: RenderPass<PushConstants>,
	late_sw: ComputePass<PushConstants>,
}

impl Passes {
	fn execute(&self, mesh: &ext::mesh_shader::Device, mut pass: PassContext, io: PassIO) {
		let visbuffer = pass.get(io.visbuffer);
		let queue = pass.get(io.queue);

		let push = PushConstants {
			instances: pass.get(io.instances).ptr(),
			camera: pass.get(io.camera).ptr(),
			queue: pass.get(io.queue).ptr(),
			stats: pass.get(io.stats).ptr(),
			output: visbuffer.storage_id.unwrap(),
			debug: io.debug.map(|d| d.get(&mut pass)),
			_pad: 0,
		};

		unsafe {
			let pass = if io.early { &self.early_hw } else { &self.late_hw }.start_empty(
				&mut pass,
				&push,
				vk::Extent2D {
					width: visbuffer.size.width,
					height: visbuffer.size.height,
				},
			);
			mesh.cmd_draw_mesh_tasks_indirect(
				pass.pass.buf,
				queue.buffer,
				std::mem::size_of::<u32>() as u64 * 2,
				1,
				std::mem::size_of::<u32>() as u32 * 3,
			);
		}

		if io.early { &self.early_sw } else { &self.late_sw }.dispatch_indirect(
			&mut pass,
			&push,
			io.queue,
			std::mem::size_of::<u32>() * 6,
		);
	}

	unsafe fn destroy(self) { unsafe {
		self.early_hw.destroy();
		self.early_sw.destroy();
		self.late_hw.destroy();
		self.late_sw.destroy();
	}}
}

impl VisBuffer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			setup: Setup::new(),
			early_instance_cull: InstanceCull::new(device, true)?,
			late_instance_cull: InstanceCull::new(device, false)?,
			early_bvh_cull: BvhCull::new(device, true)?,
			late_bvh_cull: BvhCull::new(device, false)?,
			early_meshlet_cull: MeshletCull::new(device, true)?,
			late_meshlet_cull: MeshletCull::new(device, false)?,
			hzb_gen: HzbGen::new(device)?,
			no_debug: Passes {
				early_hw: Self::hw(device, true, false)?,
				early_sw: Self::sw(device, true, false)?,
				late_hw: Self::hw(device, false, false)?,
				late_sw: Self::sw(device, false, false)?,
			},
			debug: Passes {
				early_hw: Self::hw(device, true, true)?,
				early_sw: Self::sw(device, true, true)?,
				late_hw: Self::hw(device, false, true)?,
				late_sw: Self::sw(device, false, true)?,
			},
			mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
		})
	}

	fn spec(debug: bool, early: bool) -> &'static [&'static str] {
		if debug {
			if early {
				&["passes.mesh.debug", "passes.mesh.early"]
			} else {
				&["passes.mesh.debug", "passes.mesh.late"]
			}
		} else if early {
  				&["passes.mesh.early"]
  			} else {
  				&["passes.mesh.late"]
  			}
	}

	fn hw(device: &Device, early: bool, debug: bool) -> Result<RenderPass<PushConstants>> {
		RenderPass::new(
			device,
			GraphicsPipelineDesc {
				shaders: &[
					ShaderInfo {
						shader: "passes.mesh.mesh.hw",
						spec: Self::spec(debug, early),
					},
					ShaderInfo {
						shader: "passes.mesh.pixel.main",
						spec: if debug { &["passes.mesh.debug"] } else { &[] },
					},
				],
				..Default::default()
			},
			true,
		)
	}

	fn sw(device: &Device, early: bool, debug: bool) -> Result<ComputePass<PushConstants>> {
		ComputePass::new(
			device,
			ShaderInfo {
				shader: "passes.mesh.mesh.sw",
				spec: Self::spec(debug, early),
			},
		)
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, rend: &mut WorldRenderer<'pass, '_>, info: RenderInfo,
	) -> RenderOutput {
		frame.start_region("visbuffer");

		let rstats = self.setup.stats;
		let res = self.setup.run(frame, rend, &info, self.hzb_gen.sampler());

		frame.start_region("early pass");
		frame.start_region("cull");
		self.early_instance_cull.run(frame, &res);
		self.early_bvh_cull.run(frame, &res);
		self.early_meshlet_cull.run(frame, &res);
		frame.end_region();

		let mut pass = frame.pass("rasterize");
		let instances = res.instances_mesh(&mut pass);
		let camera = res.camera_mesh(&mut pass);
		let queue = res.mesh(&mut pass);
		let stats = res.stats_mesh(&mut pass);
		let visbuffer = res.visbuffer(&mut pass);
		let debug = res.debug(&mut pass);
		let mut io = PassIO {
			early: true,
			instances,
			camera,
			queue,
			stats,
			visbuffer,
			debug,
		};
		let p = if io.debug.is_some() {
			&self.debug
		} else {
			&self.no_debug
		};
		let mesh = &self.mesh;
		pass.build(move |pass| p.execute(mesh, pass, io));

		let mut pass = frame.pass("zero render queue");
		let zero = res.mesh_zero(&mut pass);
		pass.build(move |mut pass| {
			pass.update_buffer(zero, std::mem::size_of::<u32>() * 2, &[0u32, 0, 0, 0, 0]);
		});
		frame.end_region();

		self.hzb_gen.run(frame, visbuffer, res.hzb);
		frame.start_region("late pass");
		frame.start_region("cull");
		self.late_instance_cull.run(frame, &res);
		self.late_bvh_cull.run(frame, &res);
		self.late_meshlet_cull.run(frame, &res);
		frame.end_region();

		let mut pass = frame.pass("rasterize");
		res.camera_mesh(&mut pass);
		res.mesh(&mut pass);
		res.stats_mesh(&mut pass);
		res.visbuffer(&mut pass);
		res.debug(&mut pass);
		io.early = false;
		pass.build(move |pass| p.execute(mesh, pass, io));
		frame.end_region();

		self.hzb_gen.run(frame, visbuffer, res.hzb);

		frame.end_region();
		RenderOutput {
			stats: rstats,
			instances: res.scene.instances,
			camera,
			reader: VisBufferReader {
				visbuffer,
				queue,
				debug,
			},
		}
	}

	pub unsafe fn destroy(self) { unsafe {
		self.early_instance_cull.destroy();
		self.late_instance_cull.destroy();
		self.early_bvh_cull.destroy();
		self.late_bvh_cull.destroy();
		self.early_meshlet_cull.destroy();
		self.late_meshlet_cull.destroy();
		self.hzb_gen.destroy();
		self.no_debug.destroy();
		self.debug.destroy();
	}}
}
