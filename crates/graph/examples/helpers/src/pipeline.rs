use std::ffi::CStr;

use naga::{
	back::{spv, spv::PipelineOptions},
	front::wgsl,
	valid,
	valid::{Capabilities, ValidationFlags},
	ShaderStage,
};
use radiance_graph::{
	ash::vk::{
		BlendFactor,
		BlendOp,
		ColorComponentFlags,
		CullModeFlags,
		DynamicState,
		Format,
		FrontFace,
		GraphicsPipelineCreateInfo,
		Pipeline,
		PipelineCache,
		PipelineColorBlendAttachmentState,
		PipelineColorBlendStateCreateInfo,
		PipelineDynamicStateCreateInfo,
		PipelineInputAssemblyStateCreateInfo,
		PipelineLayout,
		PipelineLayoutCreateInfo,
		PipelineMultisampleStateCreateInfo,
		PipelineRasterizationStateCreateInfo,
		PipelineRenderingCreateInfo,
		PipelineShaderStageCreateInfo,
		PipelineVertexInputStateCreateInfo,
		PipelineViewportStateCreateInfo,
		PolygonMode,
		PrimitiveTopology,
		PushConstantRange,
		Rect2D,
		SampleCountFlags,
		ShaderModuleCreateInfo,
		ShaderStageFlags,
		Viewport,
	},
	device::Device,
};

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
	device: &Device, vertex: &[u32], fragment: &[u32], format: Format, push_constants: &[PushConstantRange],
) -> (Pipeline, PipelineLayout) {
	unsafe {
		let vertex = device
			.device()
			.create_shader_module(&ShaderModuleCreateInfo::builder().code(vertex), None)
			.unwrap();
		let fragment = device
			.device()
			.create_shader_module(&ShaderModuleCreateInfo::builder().code(fragment), None)
			.unwrap();

		let layout = device
			.device()
			.create_pipeline_layout(
				&PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.base_descriptors().layout()])
					.push_constant_ranges(push_constants),
				None,
			)
			.unwrap();

		let ret = device
			.device()
			.create_graphics_pipelines(
				PipelineCache::null(),
				&[GraphicsPipelineCreateInfo::builder()
					.stages(&[
						PipelineShaderStageCreateInfo::builder()
							.stage(ShaderStageFlags::VERTEX)
							.module(vertex)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
						PipelineShaderStageCreateInfo::builder()
							.stage(ShaderStageFlags::FRAGMENT)
							.module(fragment)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
					])
					.vertex_input_state(&PipelineVertexInputStateCreateInfo::builder())
					.input_assembly_state(
						&PipelineInputAssemblyStateCreateInfo::builder().topology(PrimitiveTopology::TRIANGLE_LIST),
					)
					.viewport_state(
						&PipelineViewportStateCreateInfo::builder()
							.viewports(&[Viewport::builder().build()])
							.scissors(&[Rect2D::builder().build()]),
					)
					.rasterization_state(
						&PipelineRasterizationStateCreateInfo::builder()
							.polygon_mode(PolygonMode::FILL)
							.front_face(FrontFace::COUNTER_CLOCKWISE)
							.cull_mode(CullModeFlags::NONE)
							.line_width(1.0),
					)
					.multisample_state(
						&PipelineMultisampleStateCreateInfo::builder().rasterization_samples(SampleCountFlags::TYPE_1),
					)
					.color_blend_state(
						&PipelineColorBlendStateCreateInfo::builder().attachments(&[
							PipelineColorBlendAttachmentState::builder()
								.color_write_mask(
									ColorComponentFlags::R
										| ColorComponentFlags::G | ColorComponentFlags::B
										| ColorComponentFlags::A,
								)
								.blend_enable(true)
								.src_color_blend_factor(BlendFactor::SRC_ALPHA)
								.dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
								.color_blend_op(BlendOp::ADD)
								.src_alpha_blend_factor(BlendFactor::ONE)
								.dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
								.alpha_blend_op(BlendOp::ADD)
								.build(),
						]),
					)
					.dynamic_state(
						&PipelineDynamicStateCreateInfo::builder()
							.dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]),
					)
					.layout(layout)
					.push_next(&mut PipelineRenderingCreateInfo::builder().color_attachment_formats(&[format]))
					.build()],
				None,
			)
			.unwrap()[0];

		device.device().destroy_shader_module(vertex, None);
		device.device().destroy_shader_module(fragment, None);

		(ret, layout)
	}
}
