use ash::{ext, vk};
use bytemuck::{bytes_of, NoUninit};
use radiance_graph::{
	device::{descriptor::StorageImageId, Device, GraphicsPipelineDesc, Pipeline, ShaderInfo},
	graph::{Frame, PassContext, Res},
	resource::{BufferHandle, GpuPtr, ImageView},
	Result,
};
use vek::{Mat4, Vec2, Vec4};

pub use crate::mesh::setup::{DebugRes, DebugResId};
use crate::{
	asset::{
		rref::RRef,
		scene::{GpuInstance, Scene},
	},
	mesh::{bvh::BvhCull, hzb::HzbGen, instance::InstanceCull, meshlet::MeshletCull, setup::Setup},
};

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

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub camera: Camera,
	pub size: Vec2<u32>,
	pub debug_info: bool,
}

#[derive(Copy, Clone)]
pub struct RenderOutput {
	pub instances: GpuPtr<GpuInstance>,
	pub camera: Res<BufferHandle>,
	pub visbuffer: Res<ImageView>,
	pub debug: Option<DebugRes>,
	pub early: Res<BufferHandle>,
	pub late: Res<BufferHandle>,
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
	layout: vk::PipelineLayout,
	no_debug: [Pipeline; 2],
	debug: [Pipeline; 2],
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
		let h = (camera.fov / 2.0).tan().recip();
		let w = h / aspect;
		let near = camera.near;
		let proj = Mat4::new(
			w, 0.0, 0.0, 0.0, //
			0.0, h, 0.0, 0.0, //
			0.0, 0.0, 0.0, near, //
			0.0, 0.0, 1.0, 0.0, //
		);
		let view = camera.view;
		let view_proj = proj * view;

		Self {
			view,
			view_proj,
			h,
			near,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	early: GpuPtr<u8>,
	late: GpuPtr<u8>,
	output: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	early: bool,
	instances: GpuPtr<GpuInstance>,
	meshlets: [Res<BufferHandle>; 2],
	camera: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	debug: Option<DebugRes>,
}

impl VisBuffer {
	pub fn new(device: &Device) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
						.size(std::mem::size_of::<PushConstants>() as _)]),
				None,
			)?;

			Ok(Self {
				setup: Setup::new(),
				early_instance_cull: InstanceCull::new(device, true)?,
				late_instance_cull: InstanceCull::new(device, false)?,
				early_bvh_cull: BvhCull::new(device, true)?,
				late_bvh_cull: BvhCull::new(device, false)?,
				early_meshlet_cull: MeshletCull::new(device, true)?,
				late_meshlet_cull: MeshletCull::new(device, false)?,
				hzb_gen: HzbGen::new(device)?,
				layout,
				no_debug: [
					Self::pipeline(device, layout, true, false)?,
					Self::pipeline(device, layout, false, false)?,
				],
				debug: [
					Self::pipeline(device, layout, true, true)?,
					Self::pipeline(device, layout, false, true)?,
				],
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	fn pipeline(device: &Device, layout: vk::PipelineLayout, early: bool, debug: bool) -> Result<Pipeline> {
		device.graphics_pipeline(GraphicsPipelineDesc {
			shaders: &[
				ShaderInfo {
					shader: "passes.mesh.mesh.main",
					spec: if debug {
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
					},
				},
				ShaderInfo {
					shader: "passes.mesh.pixel.main",
					spec: if debug { &["passes.mesh.debug"] } else { &[] },
				},
			],
			layout,
			..Default::default()
		})
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
		let meshlets = res.mesh(&mut pass);
		let visbuffer = res.visbuffer(&mut pass);
		let debug = res.debug(&mut pass);
		let mut io = PassIO {
			early: true,
			instances: info.scene.instances(),
			meshlets,
			camera,
			visbuffer,
			debug,
		};
		pass.build(move |ctx| this.execute(ctx, io));
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
			visbuffer,
			early: io.meshlets[0],
			late: io.meshlets[1],
			debug,
		}
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let read = pass.get(if io.early { io.meshlets[0] } else { io.meshlets[1] });
		let visbuffer = pass.get(io.visbuffer);
		unsafe {
			let area = vk::Rect2D::default().extent(vk::Extent2D {
				width: visbuffer.size.width,
				height: visbuffer.size.height,
			});
			dev.cmd_begin_rendering(buf, &vk::RenderingInfo::default().render_area(area).layer_count(1));
			let height = visbuffer.size.height as f32;
			dev.cmd_set_viewport(
				buf,
				0,
				&[vk::Viewport {
					x: 0.0,
					y: height,
					width: visbuffer.size.width as f32,
					height: -height,
					min_depth: 0.0,
					max_depth: 1.0,
				}],
			);
			dev.cmd_set_scissor(buf, 0, &[area]);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT,
				0,
				bytes_of(&PushConstants {
					instances: io.instances,
					camera: pass.get(io.camera).ptr(),
					early: pass.get(io.meshlets[0]).ptr(),
					late: pass.get(io.meshlets[1]).ptr(),
					output: visbuffer.storage_id.unwrap(),
					debug: io.debug.map(|d| d.get(&mut pass)),
					_pad: 0,
				}),
			);

			dev.cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				if io.debug.is_some() {
					if io.early {
						self.debug[0].get()
					} else {
						self.debug[1].get()
					}
				} else {
					if io.early {
						self.no_debug[0].get()
					} else {
						self.no_debug[1].get()
					}
				},
			);
			self.mesh
				.cmd_draw_mesh_tasks_indirect(buf, read.buffer, 0, 1, std::mem::size_of::<u32>() as u32 * 3);

			dev.cmd_end_rendering(buf);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.setup.destroy(device);
		self.early_instance_cull.destroy(device);
		self.late_instance_cull.destroy(device);
		self.early_bvh_cull.destroy(device);
		self.late_bvh_cull.destroy(device);
		self.early_meshlet_cull.destroy(device);
		self.late_meshlet_cull.destroy(device);
		self.hzb_gen.destroy(device);
		let [p0, p1] = self.no_debug;
		p0.destroy();
		p1.destroy();
		let [p0, p1] = self.debug;
		p0.destroy();
		p1.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
