use ash::vk;
use bytemuck::NoUninit;
use radiance_asset_runtime::{Meshlet, Scene};
use radiance_core::{CoreDevice, CoreFrame, CorePass, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	graph::{BufferUsage, BufferUsageType, GpuBufferDesc, ReadId, Shader, WriteId},
	resource::{GpuBufferHandle, Resource},
	sync::{GlobalBarrier, UsageType},
	Result,
};
use radiance_shader_compiler::c_str;

pub struct Cull {
	pipeline: vk::Pipeline,
	layout: vk::PipelineLayout,
}

pub struct CullOutput {
	pub commands: ReadId<GpuBufferHandle>,
	pub draw_count: ReadId<GpuBufferHandle>,
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
	commands: BufferId,
	util: BufferId,
	meshlet_count: u32,
}

struct PassIO {
	meshlets: BufferId,
	commands: WriteId<GpuBufferHandle>,
	draw_count: WriteId<GpuBufferHandle>,
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

	pub fn run<'pass>(&'pass self, frame: &mut CoreFrame<'pass, '_>, scene: &'pass Scene) -> CullOutput {
		let mut pass = frame.pass("cull");
		let meshlet_count = scene.meshlets.len() / std::mem::size_of::<Meshlet>() as u64;
		let (read_c, write_c) = pass.output(
			GpuBufferDesc {
				size: meshlet_count * std::mem::size_of::<Command>() as u64,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageWrite(Shader::Compute)],
			},
		);
		let (read_u, write_u) = pass.output(
			GpuBufferDesc {
				size: std::mem::size_of::<u32>() as u64,
			},
			BufferUsage {
				usages: &[
					BufferUsageType::TransferWrite,
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);

		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					meshlets: scene.meshlets.inner().handle().id.unwrap(),
					commands: write_c,
					draw_count: write_u,
					meshlet_count: meshlet_count as u32,
				},
			)
		});

		CullOutput {
			commands: read_c,
			draw_count: read_u,
		}
	}

	fn execute(&self, mut ctx: CorePass, io: PassIO) {
		let commands = ctx.write(io.commands);
		let draw_count = ctx.write(io.draw_count);

		unsafe {
			let dev = ctx.device.device();
			let buf = ctx.buf;

			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytemuck::bytes_of(&PushConstants {
					meshlets: io.meshlets,
					commands: commands.id.unwrap(),
					util: draw_count.id.unwrap(),
					meshlet_count: io.meshlet_count,
				}),
			);
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[ctx.device.descriptors().set()],
				&[],
			);

			dev.cmd_fill_buffer(buf, draw_count.buffer, 0, 4, 0);
			dev.cmd_pipeline_barrier2(
				buf,
				&vk::DependencyInfo::builder().memory_barriers(&[GlobalBarrier {
					previous_usages: &[UsageType::TransferWrite],
					next_usages: &[
						UsageType::ShaderStorageRead(Shader::Compute),
						UsageType::ShaderStorageWrite(Shader::Compute),
					],
				}
				.into()]),
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
