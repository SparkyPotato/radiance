use std::ffi::CStr;

use naga::{
	back::{spv, spv::PipelineOptions},
	front::wgsl,
	valid,
	valid::{Capabilities, ValidationFlags},
	ShaderStage,
};
use radiance_graph::{ash::vk, device::Device};

// We use WGSL because there's a nice compiler in rust for it (and I totally didn't contribute to it).
pub fn compile(shader: &str, stage: ShaderStage) -> Vec<u32> {
	let module = wgsl::parse_str(shader).map_err(|x| x.emit_to_stderr(shader)).unwrap();
	let info = valid::Validator::new(ValidationFlags::all(), Capabilities::all())
		.validate(&module)
		.unwrap();
	spv::write_vec(
		&module,
		&info,
		&Default::default(),
		Some(&PipelineOptions {
			entry_point: "main".into(),
			shader_stage: stage,
		}),
	)
	.unwrap()
}

pub fn simple(
	device: &Device, vertex: &[u32], fragment: &[u32], format: vk::Format, push_constants: &[vk::PushConstantRange],
) -> (vk::Pipeline, vk::PipelineLayout) {
	unsafe {
		let vertex = device
			.device()
			.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(vertex), None)
			.unwrap();
		let fragment = device
			.device()
			.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(fragment), None)
			.unwrap();

		let layout = device
			.device()
			.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(push_constants),
				None,
			)
			.unwrap();

		let ret = device
			.device()
			.create_graphics_pipelines(
				vk::PipelineCache::null(),
				&[vk::GraphicsPipelineCreateInfo::builder()
					.stages(&[
						vk::PipelineShaderStageCreateInfo::builder()
							.stage(vk::ShaderStageFlags::VERTEX)
							.module(vertex)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
						vk::PipelineShaderStageCreateInfo::builder()
							.stage(vk::ShaderStageFlags::FRAGMENT)
							.module(fragment)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
					])
					.vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::builder())
					.input_assembly_state(
						&vk::PipelineInputAssemblyStateCreateInfo::builder()
							.topology(vk::PrimitiveTopology::TRIANGLE_LIST),
					)
					.viewport_state(
						&vk::PipelineViewportStateCreateInfo::builder()
							.viewports(&[vk::Viewport::builder().build()])
							.scissors(&[vk::Rect2D::builder().build()]),
					)
					.rasterization_state(
						&vk::PipelineRasterizationStateCreateInfo::builder()
							.polygon_mode(vk::PolygonMode::FILL)
							.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
							.cull_mode(vk::CullModeFlags::NONE)
							.line_width(1.0),
					)
					.multisample_state(
						&vk::PipelineMultisampleStateCreateInfo::builder()
							.rasterization_samples(vk::SampleCountFlags::TYPE_1),
					)
					.color_blend_state(
						&vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
							vk::PipelineColorBlendAttachmentState::builder()
								.color_write_mask(
									vk::ColorComponentFlags::R
										| vk::ColorComponentFlags::G | vk::ColorComponentFlags::B
										| vk::ColorComponentFlags::A,
								)
								.blend_enable(true)
								.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
								.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
								.color_blend_op(vk::BlendOp::ADD)
								.src_alpha_blend_factor(vk::BlendFactor::ONE)
								.dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
								.alpha_blend_op(vk::BlendOp::ADD)
								.build(),
						]),
					)
					.dynamic_state(
						&vk::PipelineDynamicStateCreateInfo::builder()
							.dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
					)
					.layout(layout)
					.push_next(&mut vk::PipelineRenderingCreateInfo::builder().color_attachment_formats(&[format]))
					.build()],
				None,
			)
			.unwrap()[0];

		device.device().destroy_shader_module(vertex, None);
		device.device().destroy_shader_module(fragment, None);

		(ret, layout)
	}
}
