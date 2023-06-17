use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use radiance_asset_runtime::{MeshletPointer, Scene};
use radiance_core::{CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	graph::{BufferUsage, BufferUsageType, GpuBufferDesc, ReadId, Shader, UploadBufferDesc, WriteId},
	resource::{GpuBufferHandle, Resource, UploadBufferHandle},
	Result,
};
use radiance_shader_compiler::c_str;
use vek::Mat4;

pub struct Cull {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
}

pub struct CullOutput {
	pub commands: ReadId<GpuBufferHandle>,
	pub draw_count: ReadId<GpuBufferHandle>,
	pub camera: ReadId<UploadBufferHandle>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
pub struct Command {
	index_count: u32,
	instance_count: u32,
	first_index: u32,
	vertex_offset: i32,
	first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	meshlets: BufferId,
	instances: BufferId,
	meshlet_pointers: BufferId,
	commands: BufferId,
	util: BufferId,
	camera: BufferId,
	meshlet_count: u32,
}

struct PassIO {
	meshlets: BufferId,
	instances: BufferId,
	meshlet_pointers: BufferId,
	commands: WriteId<GpuBufferHandle>,
	draw_count: ReadId<GpuBufferHandle>,
	camera: WriteId<UploadBufferHandle>,
	camera_mat: Mat4<f32>,
	meshlet_count: u32,
}

impl Cull {
	pub fn new(device: &CoreDevice, core: &RenderCore) -> Result<Self> {
		let layout = unsafe {
			device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::builder()
						.stage_flags(vk::ShaderStageFlags::COMPUTE)
						.size(std::mem::size_of::<PushConstants>() as u32)
						.build()]),
				None,
			)?
		};
		let pipeline = core.compute_pipeline(
			device,
			layout,
			core.shaders
				.shader(c_str!("radiance-passes/mesh/cull"), vk::ShaderStageFlags::COMPUTE, None),
		)?;
		Ok(Self { layout, pipeline })
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, scene: &'pass Scene, camera_viewproj: Mat4<f32>,
	) -> CullOutput {
		let mut pass = frame.pass("clear buffer");
		let (util_r, util_w) = pass.output(
			GpuBufferDesc {
				size: std::mem::size_of::<u32>() as u64,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |mut ctx| unsafe {
			let buf = ctx.write(util_w);
			ctx.device.device().cmd_fill_buffer(ctx.buf, buf.buffer, 0, 4, 0);
		});

		let mut pass = frame.pass("cull");

		pass.input(
			util_r,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);

		let meshlet_count = scene.meshlet_pointers.len() / std::mem::size_of::<MeshletPointer>() as u64;
		let (command_r, command_w) = pass.output(
			GpuBufferDesc {
				size: meshlet_count * std::mem::size_of::<Command>() as u64,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageWrite(Shader::Compute)],
			},
		);
		let (camera_r, camera_w) = pass.output(
			UploadBufferDesc {
				size: std::mem::size_of::<Mat4<f32>>() as u64,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);

		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					meshlets: scene.meshlets.inner().handle().id.unwrap(),
					instances: scene.instances.inner().handle().id.unwrap(),
					meshlet_pointers: scene.meshlet_pointers.inner().handle().id.unwrap(),
					commands: command_w,
					draw_count: util_r,
					meshlet_count: meshlet_count as u32,
					camera: camera_w,
					camera_mat: camera_viewproj,
				},
			)
		});

		CullOutput {
			commands: command_r,
			draw_count: util_r,
			camera: camera_r,
		}
	}

	fn execute(&self, mut pass: CorePass, io: PassIO) {
		let commands = pass.write(io.commands);
		let draw_count = pass.read(io.draw_count);
		let mut camera = pass.write(io.camera);

		unsafe {
			let dev = pass.device.device();
			let buf = pass.buf;
			camera.data.as_mut().copy_from_slice(bytes_of(&io.camera_mat.cols));

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					meshlets: io.meshlets,
					instances: io.instances,
					meshlet_pointers: io.meshlet_pointers,
					commands: commands.id.unwrap(),
					util: draw_count.id.unwrap(),
					meshlet_count: io.meshlet_count,
					camera: camera.id.unwrap(),
				}),
			);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);

			let workgroups = (io.meshlet_count + 63) / 64;
			if workgroups != 0 {
				dev.cmd_dispatch(buf, workgroups, 1, 1);
			}
		}
	}

	/// # Safety
	/// Appropriate synchronization must performed.
	pub unsafe fn destroy(self, device: &CoreDevice) {
		device.device().destroy_pipeline(self.pipeline, None);
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
