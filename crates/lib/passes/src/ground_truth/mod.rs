use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_asset_runtime::{rref::RRef, scene::Scene};
use radiance_core::{CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::{ASId, BufferId, StorageImageId},
	graph::{BufferUsage, BufferUsageType, ImageDesc, ImageUsage, ImageUsageType, ReadId, UploadBufferDesc, WriteId},
	resource::{BufferDesc, GpuBuffer, ImageView, Resource, UploadBufferHandle},
	sync::Shader,
	Result,
};
use radiance_shader_compiler::c_str;
use vek::{Mat4, Vec2};

use crate::mesh::visbuffer::Camera;

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub camera: Camera,
	pub size: Vec2<u32>,
}

pub struct GroundTruth {
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	sbt: GpuBuffer,
	rgen: vk::StridedDeviceAddressRegionKHR,
	miss: vk::StridedDeviceAddressRegionKHR,
	hit: vk::StridedDeviceAddressRegionKHR,
}

struct PassIO {
	write: WriteId<ImageView>,
	camera: WriteId<UploadBufferHandle>,
	info: RenderInfo,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	out: StorageImageId,
	camera: BufferId,
	instances: BufferId,
	tlas: ASId,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct CameraData {
	view: Mat4<f32>,
	proj: Mat4<f32>,
}

fn align_up(size: u64, align: u64) -> u64 { (size + align - 1) & !(align - 1) }

impl GroundTruth {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::builder()
						.stage_flags(
							vk::ShaderStageFlags::RAYGEN_KHR
								| vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
						)
						.size(std::mem::size_of::<PushConstants>() as u32)
						.build()]),
				None,
			)?;

			let pipeline = device.rt_ext().create_ray_tracing_pipelines(
				vk::DeferredOperationKHR::null(),
				core.cache.cache(),
				&[vk::RayTracingPipelineCreateInfoKHR::builder()
					.flags(vk::PipelineCreateFlags::empty())
					.stages(&[
						core.shaders
							.shader(
								c_str!("radiance-passes/ground_truth/gen"),
								vk::ShaderStageFlags::RAYGEN_KHR,
								None,
							)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/ground_truth/miss"),
								vk::ShaderStageFlags::MISS_KHR,
								None,
							)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/ground_truth/shadow"),
								vk::ShaderStageFlags::MISS_KHR,
								None,
							)
							.build(),
						core.shaders
							.shader(
								c_str!("radiance-passes/ground_truth/hit"),
								vk::ShaderStageFlags::CLOSEST_HIT_KHR,
								None,
							)
							.build(),
					])
					.groups(&[
						vk::RayTracingShaderGroupCreateInfoKHR::builder()
							.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
							.general_shader(0)
							.closest_hit_shader(vk::SHADER_UNUSED_KHR)
							.any_hit_shader(vk::SHADER_UNUSED_KHR)
							.intersection_shader(vk::SHADER_UNUSED_KHR)
							.build(),
						vk::RayTracingShaderGroupCreateInfoKHR::builder()
							.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
							.general_shader(1)
							.closest_hit_shader(vk::SHADER_UNUSED_KHR)
							.any_hit_shader(vk::SHADER_UNUSED_KHR)
							.intersection_shader(vk::SHADER_UNUSED_KHR)
							.build(),
						vk::RayTracingShaderGroupCreateInfoKHR::builder()
							.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
							.general_shader(2)
							.closest_hit_shader(vk::SHADER_UNUSED_KHR)
							.any_hit_shader(vk::SHADER_UNUSED_KHR)
							.intersection_shader(vk::SHADER_UNUSED_KHR)
							.build(),
						vk::RayTracingShaderGroupCreateInfoKHR::builder()
							.ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
							.general_shader(vk::SHADER_UNUSED_KHR)
							.closest_hit_shader(3)
							.any_hit_shader(vk::SHADER_UNUSED_KHR)
							.intersection_shader(vk::SHADER_UNUSED_KHR)
							.build(),
					])
					.max_pipeline_ray_recursion_depth(2)
					.layout(layout)
					.build()],
				None,
			)?[0];

			let mut rgen = vk::StridedDeviceAddressRegionKHR::default();
			let mut miss = vk::StridedDeviceAddressRegionKHR::default();
			let mut hit = vk::StridedDeviceAddressRegionKHR::default();

			let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
			let mut p = vk::PhysicalDeviceProperties2::builder().push_next(&mut props);
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

			let sbt = GpuBuffer::create(
				device,
				BufferDesc {
					size: rgen.size + miss.size + hit.size,
					usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
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
				sbt,
				rgen,
				miss,
				hit,
			})
		}
	}

	pub fn destroy(self, device: &CoreDevice) {
		unsafe {
			self.sbt.destroy(device);
			device.device().destroy_pipeline(self.pipeline, None);
			device.device().destroy_pipeline_layout(self.layout, None);
		}
	}

	pub fn run<'pass>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, info: RenderInfo,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("raytrace");
		let (read, write) = pass.output(
			ImageDesc {
				size: vk::Extent3D {
					width: info.size.x,
					height: info.size.y,
					depth: 1,
				},
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
			},
			ImageUsage {
				format: vk::Format::R8G8B8A8_UNORM,
				usages: &[ImageUsageType::ShaderStorageWrite(Shader::RayTracing)],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);

		let (_, camera) = pass.output(
			UploadBufferDesc {
				size: std::mem::size_of::<CameraData>() as _,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::RayTracing)],
			},
		);

		pass.build(|pass| self.execute(pass, PassIO { write, camera, info }));

		read
	}

	fn execute(&self, mut pass: CorePass, io: PassIO) {
		let mut camera = pass.write(io.camera);
		let write = pass.write(io.write);

		let dev = pass.device;
		let buf = pass.buf;

		unsafe {
			let s = io.info.size.map(|x| x as f32);
			let proj =
				Mat4::perspective_fov_lh_zo(io.info.camera.fov, s.x, s.y, io.info.camera.near, 10000.0).inverted();
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
					camera: camera.id.unwrap(),
					instances: io.info.scene.instances(),
					tlas: io.info.scene.acceleration_structure(),
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

