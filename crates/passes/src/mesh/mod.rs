use ash::{ext, vk};
use bytemuck::{cast_slice, NoUninit};
use radiance_asset::{
	rref::RRef,
	scene::{GpuInstance, Scene},
};
use radiance_graph::{
	device::{descriptor::StorageImageId, Device, GraphicsPipelineDesc, ShaderInfo},
	graph::{BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, PassBuilder, PassContext, Res},
	resource::{BufferHandle, GpuPtr, ImageView, Subresource},
	sync::Shader,
	util::{compute::ComputePass, render::RenderPass},
	Result,
};
use vek::{Mat4, Vec2};

pub use crate::mesh::setup::{DebugRes, DebugResId};
use crate::mesh::{bvh::BvhCull, hzb::HzbGen, instance::InstanceCull, meshlet::MeshletCull, setup::Setup};

mod bvh;
mod hzb;
mod instance;
mod meshlet;
mod setup;

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Camera {
	/// Vertical FOV in radians.
	pub fov: f32,
	pub near: f32,
	/// View matrix (inverse of camera transform).
	pub view: Mat4<f32>,
}

impl Camera {
	pub fn projection(&self, aspect: f32) -> Mat4<f32> {
		let h = (self.fov / 2.0).tan().recip();
		let w = h / aspect;
		let near = self.near;
		Mat4::new(
			w, 0.0, 0.0, 0.0, //
			0.0, h, 0.0, 0.0, //
			0.0, 0.0, 0.0, near, //
			0.0, 0.0, 1.0, 0.0, //
		)
	}
}

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub camera: Camera,
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
		pass.reference(
			self.queue,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(shader)],
			},
		);

		let usage = ImageUsage {
			format: vk::Format::UNDEFINED,
			usages: &[ImageUsageType::ShaderStorageRead(Shader::Fragment)],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};
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
	pub instances: GpuPtr<GpuInstance>,
	pub camera: Res<BufferHandle>,
	pub reader: VisBufferReader,
}

struct Passes {
	early_hw: RenderPass<PushConstants>,
	early_sw: ComputePass<PushConstants>,
	late_hw: RenderPass<PushConstants>,
	late_sw: ComputePass<PushConstants>,
}

impl Passes {
	unsafe fn destroy(self) {
		self.early_hw.destroy();
		self.early_sw.destroy();
		self.late_hw.destroy();
		self.late_sw.destroy();
	}
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
#[derive(Copy, Clone, NoUninit)]
pub struct CameraData {
	view: Mat4<f32>,
	view_proj: Mat4<f32>,
	h: f32,
	near: f32,
}

impl CameraData {
	fn new(aspect: f32, camera: Camera) -> Self {
		let proj = camera.projection(aspect);
		let view = camera.view;
		let view_proj = proj * view;
		Self {
			view,
			view_proj,
			h: proj.cols.y.y,
			near: camera.near,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	queue: GpuPtr<u8>,
	output: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	early: bool,
	instances: GpuPtr<GpuInstance>,
	queue: Res<BufferHandle>,
	camera: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	debug: Option<DebugRes>,
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
		} else {
			if early {
				&["passes.mesh.early"]
			} else {
				&["passes.mesh.late"]
			}
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

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> RenderOutput {
		frame.start_region("visbuffer");

		let res = self.setup.run(frame, &info, self.hzb_gen.sampler());
		let this: &Self = self;

		frame.start_region("early pass");
		frame.start_region("cull");
		this.early_instance_cull.run(frame, &info, &res);
		this.early_bvh_cull.run(frame, &info, &res);
		this.early_meshlet_cull.run(frame, &info, &res);
		frame.end_region();

		let mut pass = frame.pass("rasterize");
		let camera = res.camera_mesh(&mut pass);
		let queue = res.mesh(&mut pass);
		let visbuffer = res.visbuffer(&mut pass);
		let debug = res.debug(&mut pass);
		let mut io = PassIO {
			early: true,
			instances: info.scene.instances(),
			queue,
			camera,
			visbuffer,
			debug,
		};
		pass.build(move |ctx| this.execute(ctx, io));

		let mut pass = frame.pass("zero render queue");
		let zero = res.mesh_zero(&mut pass);
		pass.build(move |mut ctx| unsafe {
			let zero = ctx.get(zero);
			ctx.device.device().cmd_update_buffer(
				ctx.buf,
				zero.buffer,
				std::mem::size_of::<u32>() as u64 * 2,
				cast_slice(&[0u32]),
			);
			ctx.device.device().cmd_update_buffer(
				ctx.buf,
				zero.buffer,
				std::mem::size_of::<u32>() as u64 * 6,
				cast_slice(&[0u32]),
			);
		});
		frame.end_region();

		this.hzb_gen.run(frame, visbuffer, res.hzb);
		frame.start_region("late pass");
		frame.start_region("cull");
		this.late_instance_cull.run(frame, &info, &res);
		this.late_bvh_cull.run(frame, &info, &res);
		this.late_meshlet_cull.run(frame, &info, &res);
		frame.end_region();

		let mut pass = frame.pass("rasterize");
		res.camera_mesh(&mut pass);
		res.mesh(&mut pass);
		res.visbuffer(&mut pass);
		res.debug(&mut pass);
		io.early = false;
		pass.build(move |ctx| this.execute(ctx, io));
		frame.end_region();

		this.hzb_gen.run(frame, visbuffer, res.hzb);

		frame.end_region();
		RenderOutput {
			instances: info.scene.instances(),
			camera,
			reader: VisBufferReader {
				visbuffer,
				queue,
				debug,
			},
		}
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let visbuffer = pass.get(io.visbuffer);
		let queue = pass.get(io.queue);

		let push = PushConstants {
			instances: io.instances,
			camera: pass.get(io.camera).ptr(),
			queue: pass.get(io.queue).ptr(),
			output: visbuffer.storage_id.unwrap(),
			debug: io.debug.map(|d| d.get(&mut pass)),
			_pad: 0,
		};

		if io.debug.is_some() {
			if io.early {
				&self.debug.early_hw
			} else {
				&self.debug.late_hw
			}
		} else {
			if io.early {
				&self.no_debug.early_hw
			} else {
				&self.no_debug.late_hw
			}
		}
		.run_empty(
			&pass,
			&push,
			vk::Extent2D {
				width: visbuffer.size.width,
				height: visbuffer.size.height,
			},
			|_, buf| unsafe {
				self.mesh.cmd_draw_mesh_tasks_indirect(
					buf,
					queue.buffer,
					std::mem::size_of::<u32>() as u64 * 2,
					1,
					std::mem::size_of::<u32>() as u32 * 3,
				);
			},
		);

		if io.debug.is_some() {
			if io.early {
				&self.debug.early_sw
			} else {
				&self.debug.late_sw
			}
		} else {
			if io.early {
				&self.no_debug.early_sw
			} else {
				&self.no_debug.late_sw
			}
		}
		.dispatch_indirect(&push, &pass, queue.buffer, std::mem::size_of::<u32>() * 6);
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.early_instance_cull.destroy();
		self.late_instance_cull.destroy();
		self.early_bvh_cull.destroy();
		self.late_bvh_cull.destroy();
		self.early_meshlet_cull.destroy();
		self.late_meshlet_cull.destroy();
		self.hzb_gen.destroy(device);
		self.no_debug.destroy();
		self.debug.destroy();
	}
}
