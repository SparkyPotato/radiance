use ash::{ext, vk};
use bytemuck::{bytes_of, NoUninit};
use radiance_asset::{
	rref::RRef,
	scene::{GpuInstance, Scene},
};
use radiance_graph::{
	device::{descriptor::StorageImageId, Device, GraphicsPipelineDesc, Pipeline, ShaderInfo},
	graph::{BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, PassBuilder, PassContext, Res},
	resource::{BufferHandle, GpuPtr, ImageView, Subresource},
	sync::Shader,
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
	pub early_hw: Res<BufferHandle>,
	pub early_sw: Res<BufferHandle>,
	pub late_hw: Res<BufferHandle>,
	pub late_sw: Res<BufferHandle>,
	pub debug: Option<DebugRes>,
}

impl VisBufferReader {
	pub fn add(&self, pass: &mut PassBuilder, shader: Shader, debug: bool) {
		let usage = BufferUsage {
			usages: &[BufferUsageType::ShaderStorageRead(shader)],
		};
		pass.reference(self.early_hw, usage);
		pass.reference(self.early_sw, usage);
		pass.reference(self.late_hw, usage);
		pass.reference(self.late_sw, usage);

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
			early_hw: pass.get(self.early_hw).ptr(),
			early_sw: pass.get(self.early_sw).ptr(),
			late_hw: pass.get(self.late_hw).ptr(),
			late_sw: pass.get(self.late_sw).ptr(),
			visbuffer: pass.get(self.visbuffer).storage_id.unwrap(),
			_pad: 0,
		}
	}

	pub fn get_debug(self, pass: &mut PassContext) -> GpuVisBufferReaderDebug {
		GpuVisBufferReaderDebug {
			early_hw: pass.get(self.early_hw).ptr(),
			early_sw: pass.get(self.early_sw).ptr(),
			late_hw: pass.get(self.late_hw).ptr(),
			late_sw: pass.get(self.late_sw).ptr(),
			visbuffer: pass.get(self.visbuffer).storage_id.unwrap(),
			debug: self.debug.map(|x| x.get(pass)),
			_pad: 0,
		}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuVisBufferReaderDebug {
	early_hw: GpuPtr<u8>,
	early_sw: GpuPtr<u8>,
	late_hw: GpuPtr<u8>,
	late_sw: GpuPtr<u8>,
	visbuffer: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuVisBufferReader {
	early_hw: GpuPtr<u8>,
	early_sw: GpuPtr<u8>,
	late_hw: GpuPtr<u8>,
	late_sw: GpuPtr<u8>,
	visbuffer: StorageImageId,
	_pad: u32,
}

#[derive(Copy, Clone)]
pub struct RenderOutput {
	pub instances: GpuPtr<GpuInstance>,
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
	layout: vk::PipelineLayout,
	no_debug: [Pipeline; 4],
	debug: [Pipeline; 4],
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
	early_hw: GpuPtr<u8>,
	early_sw: GpuPtr<u8>,
	late_hw: GpuPtr<u8>,
	late_sw: GpuPtr<u8>,
	output: StorageImageId,
	debug: Option<DebugResId>,
	_pad: u32,
}

#[derive(Copy, Clone)]
struct PassIO {
	early: bool,
	instances: GpuPtr<GpuInstance>,
	meshlets: [Res<BufferHandle>; 4],
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
						.stage_flags(
							vk::ShaderStageFlags::COMPUTE
								| vk::ShaderStageFlags::MESH_EXT
								| vk::ShaderStageFlags::FRAGMENT,
						)
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
					Self::pipeline(device, layout, false, true, false)?,
					Self::pipeline(device, layout, true, true, false)?,
					Self::pipeline(device, layout, false, false, false)?,
					Self::pipeline(device, layout, true, false, false)?,
				],
				debug: [
					Self::pipeline(device, layout, false, true, true)?,
					Self::pipeline(device, layout, true, true, true)?,
					Self::pipeline(device, layout, false, false, true)?,
					Self::pipeline(device, layout, true, false, true)?,
				],
				mesh: ext::mesh_shader::Device::new(device.instance(), device.device()),
			})
		}
	}

	fn pipeline(device: &Device, layout: vk::PipelineLayout, sw: bool, early: bool, debug: bool) -> Result<Pipeline> {
		let spec: &[_] = if debug {
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
		};
		if sw {
			device.compute_pipeline(
				layout,
				ShaderInfo {
					shader: "passes.mesh.mesh.sw",
					spec,
				},
			)
		} else {
			device.graphics_pipeline(GraphicsPipelineDesc {
				shaders: &[
					ShaderInfo {
						shader: "passes.mesh.mesh.hw",
						spec,
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
			reader: VisBufferReader {
				visbuffer,
				early_hw: io.meshlets[0],
				early_sw: io.meshlets[1],
				late_hw: io.meshlets[2],
				late_sw: io.meshlets[3],
				debug,
			},
		}
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let dev = pass.device.device();
		let buf = pass.buf;
		let read_hw = pass.get(if io.early { io.meshlets[0] } else { io.meshlets[2] });
		let read_sw = pass.get(if io.early { io.meshlets[1] } else { io.meshlets[3] });
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
				vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					instances: io.instances,
					camera: pass.get(io.camera).ptr(),
					early_hw: pass.get(io.meshlets[0]).ptr(),
					early_sw: pass.get(io.meshlets[1]).ptr(),
					late_hw: pass.get(io.meshlets[2]).ptr(),
					late_sw: pass.get(io.meshlets[3]).ptr(),
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
						self.debug[2].get()
					}
				} else {
					if io.early {
						self.no_debug[0].get()
					} else {
						self.no_debug[2].get()
					}
				},
			);
			self.mesh
				.cmd_draw_mesh_tasks_indirect(buf, read_hw.buffer, 0, 1, std::mem::size_of::<u32>() as u32 * 3);

			dev.cmd_end_rendering(buf);

			dev.cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				if io.debug.is_some() {
					if io.early {
						self.debug[1].get()
					} else {
						self.debug[3].get()
					}
				} else {
					if io.early {
						self.no_debug[1].get()
					} else {
						self.no_debug[3].get()
					}
				},
			);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			dev.cmd_dispatch_indirect(buf, read_sw.buffer, 0);
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.early_instance_cull.destroy(device);
		self.late_instance_cull.destroy(device);
		self.early_bvh_cull.destroy(device);
		self.late_bvh_cull.destroy(device);
		self.early_meshlet_cull.destroy(device);
		self.late_meshlet_cull.destroy(device);
		self.hzb_gen.destroy(device);
		let [p0, p1, p2, p3] = self.no_debug;
		p0.destroy();
		p1.destroy();
		p2.destroy();
		p3.destroy();
		let [p0, p1, p2, p3] = self.debug;
		p0.destroy();
		p1.destroy();
		p2.destroy();
		p3.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
