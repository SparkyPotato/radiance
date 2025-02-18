use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
		RtPipeline,
		RtPipelineDesc,
		RtShaderGroup,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{BufferUsage, Frame, ImageDesc, ImageUsage, Persist, Res},
	resource::{GpuPtr, Image, ImageView},
	sync::Shader,
	Result,
};
use rand::{thread_rng, RngCore};
use vek::Vec2;

use crate::{
	scene::{
		camera::{CameraScene, GpuCamera},
		light::{GpuLight, LightScene},
		rt_scene::{GpuRtInstance, RtScene},
		WorldRenderer,
	},
	sky::{GpuSkySampler, SkySampler},
};

// TODO: reset on world change and edit.
pub struct PathTracer {
	pipeline: RtPipeline,
	sampler: SamplerId,
	accum: Persist<Image>,
	cached: Option<Vec2<u32>>,
	samples: u32,
}

pub struct RenderInfo {
	pub sky: SkySampler,
	pub size: Vec2<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuRtInstance>,
	lights: GpuPtr<GpuLight>,
	camera: GpuPtr<GpuCamera>,
	as_: GpuPtr<u8>,
	sampler: SamplerId,
	out: StorageImageId,
	seed: u32,
	samples: u32,
	light_count: u32,
	sky: GpuSkySampler,
}

impl PathTracer {
	pub fn new(device: &Device) -> Result<Self> {
		let pipeline = device.rt_pipeline(RtPipelineDesc {
			shaders: &[
				ShaderInfo {
					shader: "passes.pt.gen.main",
					spec: &[],
				},
				ShaderInfo {
					shader: "passes.pt.miss.main",
					spec: &[],
				},
				ShaderInfo {
					shader: "passes.pt.shadow.main",
					spec: &[],
				},
				ShaderInfo {
					shader: "passes.pt.hit.main",
					spec: &[],
				},
			],
			groups: &[
				RtShaderGroup::General(0),
				RtShaderGroup::General(1),
				RtShaderGroup::General(2),
				RtShaderGroup::Triangles {
					closest_hit: Some(3),
					any_hit: None,
				},
			],
			recursion_depth: 1,
		})?;

		Ok(Self {
			pipeline,
			sampler: device.sampler(SamplerDesc::default()),
			accum: Persist::new(),
			cached: None,
			samples: 0,
		})
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, rend: &mut WorldRenderer<'pass, '_>, info: RenderInfo,
	) -> (Res<ImageView>, u32) {
		let rt = rend.get::<RtScene>(frame);
		let camera = rend.get::<CameraScene>(frame);
		let lights = rend.get::<LightScene>(frame);

		let mut pass = frame.pass("path trace");

		let read = BufferUsage::read(Shader::RayTracing);
		pass.reference(rt.instances, read);
		pass.reference(rt.as_, read);
		pass.reference(camera.buf, read);
		pass.reference(lights.buf, read);
		info.sky.reference(&mut pass, Shader::RayTracing);

		let out = pass.resource(
			ImageDesc {
				format: vk::Format::R32G32B32A32_SFLOAT,
				size: vk::Extent3D {
					width: info.size.x,
					height: info.size.y,
					depth: 1,
				},
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				persist: Some(self.accum),
			},
			ImageUsage::read_write_2d(Shader::RayTracing),
		);

		if let Some(c) = self.cached {
			if c != info.size {
				self.samples = 0;
			}
		}
		self.cached = Some(info.size);

		let s = self.samples;
		pass.build(move |mut pass| {
			if pass.is_uninit(out) || camera.prev != camera.curr {
				self.samples = 0;
			}

			let out = pass.get(out);
			let as_ = pass.get(rt.as_).ptr().offset(rt.as_offset);
			let instances = pass.get(rt.instances).ptr();
			let light_count = lights.count;
			let lights = pass.get(lights.buf).ptr();
			let camera = pass.get(camera.buf).ptr();
			let sky = info.sky.to_gpu(&mut pass);

			self.pipeline.bind(pass.buf);
			pass.push(
				0,
				&PushConstants {
					instances,
					lights,
					camera,
					as_,
					sampler: self.sampler,
					out: out.storage_id.unwrap(),
					seed: thread_rng().next_u32(),
					samples: self.samples,
					light_count,
					sky,
				},
			);
			self.pipeline.trace_rays(pass.buf, out.size.width, out.size.height, 1);

			self.samples += 1;
		});

		(out, s)
	}

	pub unsafe fn destroy(self) { self.pipeline.destroy(); }
}
