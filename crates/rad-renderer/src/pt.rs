use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	c_str,
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{BufferUsage, Frame, ImageDesc, ImageUsage, Res},
	resource::{self, Buffer, GpuPtr, ImageView, Resource},
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
	pipeline: vk::Pipeline,
	sbt: Buffer,
	rgen: vk::StridedDeviceAddressRegionKHR,
	miss: vk::StridedDeviceAddressRegionKHR,
	hit: vk::StridedDeviceAddressRegionKHR,
	sampler: SamplerId,
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

fn align_up(size: u64, align: u64) -> u64 { (size + align - 1) & !(align - 1) }

impl PathTracer {
	pub fn new(device: &Device) -> Result<Self> {
		let rgen = device.get_shader(ShaderInfo {
			shader: "passes.pt.gen.main",
			spec: &[],
		})?;
		let miss = device.get_shader(ShaderInfo {
			shader: "passes.pt.miss.main",
			spec: &[],
		})?;
		let shadow = device.get_shader(ShaderInfo {
			shader: "passes.pt.shadow.main",
			spec: &[],
		})?;
		let hit = device.get_shader(ShaderInfo {
			shader: "passes.pt.hit.main",
			spec: &[],
		})?;
		unsafe {
			let pipeline = device
				.rt_ext()
				.create_ray_tracing_pipelines(
					vk::DeferredOperationKHR::null(),
					vk::PipelineCache::null(),
					&[vk::RayTracingPipelineCreateInfoKHR::default()
						.flags(vk::PipelineCreateFlags::empty())
						.stages(&[
							vk::PipelineShaderStageCreateInfo::default()
								.stage(rgen.1)
								.name(c_str!("main"))
								.push_next(&mut vk::ShaderModuleCreateInfo::default().code(&rgen.0)),
							vk::PipelineShaderStageCreateInfo::default()
								.stage(miss.1)
								.name(c_str!("main"))
								.push_next(&mut vk::ShaderModuleCreateInfo::default().code(&miss.0)),
							vk::PipelineShaderStageCreateInfo::default()
								.stage(shadow.1)
								.name(c_str!("main"))
								.push_next(&mut vk::ShaderModuleCreateInfo::default().code(&shadow.0)),
							vk::PipelineShaderStageCreateInfo::default()
								.stage(hit.1)
								.name(c_str!("main"))
								.push_next(&mut vk::ShaderModuleCreateInfo::default().code(&hit.0)),
						])
						.groups(&[
							vk::RayTracingShaderGroupCreateInfoKHR::default()
								.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
								.general_shader(0)
								.closest_hit_shader(vk::SHADER_UNUSED_KHR)
								.any_hit_shader(vk::SHADER_UNUSED_KHR)
								.intersection_shader(vk::SHADER_UNUSED_KHR),
							vk::RayTracingShaderGroupCreateInfoKHR::default()
								.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
								.general_shader(1)
								.closest_hit_shader(vk::SHADER_UNUSED_KHR)
								.any_hit_shader(vk::SHADER_UNUSED_KHR)
								.intersection_shader(vk::SHADER_UNUSED_KHR),
							vk::RayTracingShaderGroupCreateInfoKHR::default()
								.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
								.general_shader(2)
								.closest_hit_shader(vk::SHADER_UNUSED_KHR)
								.any_hit_shader(vk::SHADER_UNUSED_KHR)
								.intersection_shader(vk::SHADER_UNUSED_KHR),
							vk::RayTracingShaderGroupCreateInfoKHR::default()
								.ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
								.general_shader(vk::SHADER_UNUSED_KHR)
								.closest_hit_shader(3)
								.any_hit_shader(vk::SHADER_UNUSED_KHR)
								.intersection_shader(vk::SHADER_UNUSED_KHR),
						])
						.max_pipeline_ray_recursion_depth(2)
						.layout(device.layout())],
					None,
				)
				.unwrap()[0];

			let mut rgen = vk::StridedDeviceAddressRegionKHR::default();
			let mut miss = vk::StridedDeviceAddressRegionKHR::default();
			let mut hit = vk::StridedDeviceAddressRegionKHR::default();

			let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
			let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
			device
				.instance()
				.get_physical_device_properties2(device.physical_device(), &mut p);

			let handle_count = 1 + 2 + 1;
			let handle_size = props.shader_group_handle_size as u64;
			let handle_align = props.shader_group_handle_alignment as u64;
			let base_align = props.shader_group_base_alignment as u64;
			let handle_size_align = align_up(handle_size, handle_align);
			rgen.stride = align_up(handle_size_align, base_align);
			rgen.size = rgen.stride;
			miss.stride = handle_size_align;
			miss.size = align_up(2 * handle_size_align, base_align);
			hit.stride = handle_size_align;
			hit.size = align_up(1 * handle_size, base_align);

			let sbt = Buffer::create(
				device,
				resource::BufferDesc {
					name: "pt sbt",
					size: rgen.size + miss.size + hit.size,
					readback: false,
				},
			)?;
			rgen.device_address = sbt.ptr::<()>().addr();
			miss.device_address = rgen.device_address + rgen.size;
			hit.device_address = miss.device_address + miss.size;

			let handles = device.rt_ext().get_ray_tracing_shader_group_handles(
				pipeline,
				0,
				handle_count,
				(handle_count as u64 * handle_size) as usize,
			)?;

			let p = sbt.data().as_ptr().cast::<u8>();
			std::ptr::copy_nonoverlapping(handles.as_ptr(), p, handle_size as usize);
			std::ptr::copy_nonoverlapping(
				handles.as_ptr().add(1 * handle_size as usize),
				p.add(rgen.size as usize),
				handle_size as usize,
			);
			std::ptr::copy_nonoverlapping(
				handles.as_ptr().add(2 * handle_size as usize),
				p.add((rgen.size + handle_size_align) as usize),
				handle_size as usize,
			);
			std::ptr::copy_nonoverlapping(
				handles.as_ptr().add(3 * handle_size as usize),
				p.add((rgen.size + miss.size) as usize),
				handle_size as usize,
			);

			Ok(Self {
				pipeline,
				sbt,
				rgen,
				miss,
				hit,
				sampler: device.sampler(SamplerDesc::default()),
				cached: None,
				samples: 0,
			})
		}
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
				persist: Some("path tracer accum"),
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

			unsafe {
				pass.device
					.device()
					.cmd_bind_pipeline(pass.buf, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);
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
				pass.device.rt_ext().cmd_trace_rays(
					pass.buf,
					&self.rgen,
					&self.miss,
					&self.hit,
					&vk::StridedDeviceAddressRegionKHR::default(),
					out.size.width,
					out.size.height,
					1,
				);
			}

			self.samples += 1;
		});

		(out, s)
	}

	// TODO: destroy
	pub unsafe fn destroy(self) {}
}
