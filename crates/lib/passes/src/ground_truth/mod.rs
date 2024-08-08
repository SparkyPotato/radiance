use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, AnyBitPattern, NoUninit};
use radiance_graph::{
	device::{
		descriptor::{ASId, SamplerId, StorageImageId},
		Device,
	},
	graph::{
		self,
		BufferUsage,
		BufferUsageType,
		ExternalImage,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Res,
	},
	resource::{Buffer, BufferDesc, BufferHandle, Image, ImageView, Resource, Subresource},
	sync::Shader,
	Result,
};
use vek::{Mat4, Vec2};

use crate::{
	asset::{
		material::GpuMaterial,
		rref::{RRef, RWeak},
		scene::{GpuInstance, Scene},
	},
	mesh::visbuffer::Camera,
};

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub materials: *mut GpuMaterial,
	pub camera: Camera,
	pub size: Vec2<u32>,
}

pub struct GroundTruth {
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	sampler: vk::Sampler,
	sampler_id: SamplerId,
	sbt: Buffer,
	accum: Image,
	size: vk::Extent3D,
	samples: u32,
	last_cam: Camera,
	last_scene: RWeak<Scene>,
	rgen: vk::StridedDeviceAddressRegionKHR,
	miss: vk::StridedDeviceAddressRegionKHR,
	hit: vk::StridedDeviceAddressRegionKHR,
}

struct PassIO {
	write: Res<ImageView>,
	camera: Res<BufferHandle>,
	info: RenderInfo,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct PushConstants {
	out: StorageImageId,
	samples: u32,
	camera: *mut CameraData,
	instances: *mut GpuInstance,
	materials: *mut GpuMaterial,
	tlas: ASId,
	sampler: SamplerId,
	seed: u32,
}

unsafe impl NoUninit for PushConstants {}

#[derive(Copy, Clone, NoUninit, AnyBitPattern)]
#[repr(C)]
struct CameraData {
	view: Mat4<f32>,
	proj: Mat4<f32>,
}

fn align_up(size: u64, align: u64) -> u64 { (size + align - 1) & !(align - 1) }

impl GroundTruth {
	pub fn new(device: &Device) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(
							vk::ShaderStageFlags::RAYGEN_KHR
								| vk::ShaderStageFlags::MISS_KHR
								| vk::ShaderStageFlags::CLOSEST_HIT_KHR,
						)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?;

			let pipeline = device
				.rt_ext()
				.create_ray_tracing_pipelines(
					vk::DeferredOperationKHR::null(),
					vk::PipelineCache::null(),
					&[vk::RayTracingPipelineCreateInfoKHR::default()
						.flags(vk::PipelineCreateFlags::empty())
						.stages(&[
							device.shader(
								"radiance-passes/ground_truth/gen",
								vk::ShaderStageFlags::RAYGEN_KHR,
								None,
							),
							device.shader(
								"radiance-passes/ground_truth/miss",
								vk::ShaderStageFlags::MISS_KHR,
								None,
							),
							device.shader(
								"radiance-passes/ground_truth/shadow",
								vk::ShaderStageFlags::MISS_KHR,
								None,
							),
							device.shader(
								"radiance-passes/ground_truth/hit",
								vk::ShaderStageFlags::CLOSEST_HIT_KHR,
								None,
							),
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
						.layout(layout)],
					None,
				)
				.map_err(|(_, x)| x)?[0];

			let sampler = device.device().create_sampler(
				&vk::SamplerCreateInfo::default()
					.mag_filter(vk::Filter::LINEAR)
					.min_filter(vk::Filter::LINEAR)
					.mipmap_mode(vk::SamplerMipmapMode::LINEAR)
					.address_mode_u(vk::SamplerAddressMode::REPEAT)
					.address_mode_v(vk::SamplerAddressMode::REPEAT),
				None,
			)?;

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
				BufferDesc {
					name: "ground truth shader binding table",
					size: rgen.size + miss.size + hit.size,
					usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
					on_cpu: false,
				},
			)?;
			rgen.device_address = sbt.addr();
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
				layout,
				pipeline,
				sampler,
				sampler_id: device.descriptors().get_sampler(device, sampler),
				sbt,
				size: vk::Extent3D::default(),
				samples: 0,
				accum: Image::default(),
				last_cam: Camera::default(),
				last_scene: RWeak::new(),
				rgen,
				miss,
				hit,
			})
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			self.sbt.destroy(device);
			self.accum.destroy(device);
			device.descriptors().return_sampler(self.sampler_id);
			device.device().destroy_sampler(self.sampler, None);
			device.device().destroy_pipeline(self.pipeline, None);
			device.device().destroy_pipeline_layout(self.layout, None);
		}
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let size = vk::Extent3D {
			width: info.size.x,
			height: info.size.y,
			depth: 1,
		};
		let new = self.size != size;
		let w = info.scene.downgrade();
		let clear = self.last_cam != info.camera || new || !self.last_scene.ptr_eq(&w);
		self.last_cam = info.camera;
		self.last_scene = w;

		if new {
			unsafe {
				std::mem::take(&mut self.accum).destroy(frame.device());
				self.accum = Image::create(
					frame.device(),
					radiance_graph::resource::ImageDesc {
						name: "ground truth accumulation",
						flags: vk::ImageCreateFlags::empty(),
						format: vk::Format::R16G16B16A16_SFLOAT,
						size,
						levels: 1,
						layers: 1,
						samples: vk::SampleCountFlags::TYPE_1,
						usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
					},
				)
				.unwrap();
				self.size = size;
			}
		}

		self.samples = if clear { 0 } else { self.samples + 1 };

		let mut pass = frame.pass("raytrace");
		let write = pass.resource(
			ExternalImage {
				handle: self.accum.handle(),
				desc: ImageDesc {
					size,
					levels: 1,
					layers: 1,
					samples: vk::SampleCountFlags::TYPE_1,
					format: vk::Format::R16G16B16A16_SFLOAT,
				},
			},
			ImageUsage {
				format: vk::Format::R16G16B16A16_SFLOAT,
				usages: &[
					ImageUsageType::ShaderStorageRead(Shader::RayTracing),
					ImageUsageType::ShaderStorageWrite(Shader::RayTracing),
				],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		let camera = pass.resource(
			graph::BufferDesc {
				size: std::mem::size_of::<CameraData>() as _,
				upload: true,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::RayTracing)],
			},
		);

		pass.build(move |pass| self.execute(pass, PassIO { write, camera, info }));

		write
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let mut camera = pass.get(io.camera);
		let write = pass.get(io.write);

		let dev = pass.device;
		let buf = pass.buf;

		unsafe {
			let s = io.info.size.map(|x| x as f32);
			let proj = Mat4::perspective_fov_lh_zo(io.info.camera.fov, s.x, s.y, io.info.camera.near, 10.0).inverted();
			camera
				.data
				.as_mut()
				.write(bytes_of(&CameraData {
					view: io.info.camera.view.inverted(),
					proj,
				}))
				.unwrap();

			dev.device()
				.cmd_bind_pipeline(buf, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);
			dev.device().cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::RAY_TRACING_KHR,
				self.layout,
				0,
				&[dev.descriptors().set()],
				&[],
			);
			dev.device().cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::RAYGEN_KHR
					| vk::ShaderStageFlags::MISS_KHR
					| vk::ShaderStageFlags::CLOSEST_HIT_KHR,
				0,
				bytes_of(&PushConstants {
					out: write.storage_id.unwrap(),
					samples: self.samples,
					camera: camera.as_gpu(),
					instances: io.info.scene.instances(),
					materials: io.info.materials,
					tlas: io.info.scene.acceleration_structure(),
					sampler: self.sampler_id,
					seed: rand::random(),
				}),
			);

			dev.rt_ext().cmd_trace_rays(
				buf,
				&self.rgen,
				&self.miss,
				&self.hit,
				&vk::StridedDeviceAddressRegionKHR::default(),
				io.info.size.x,
				io.info.size.y,
				1,
			);
		}
	}
}
