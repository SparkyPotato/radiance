use radiance_graph::{
	ash::{vk, vk::TaggedStructure},
	Result,
};

use crate::{CoreDevice, RenderCore};

pub struct GraphicsPipelineDesc<'a> {
	pub shaders: &'a [vk::PipelineShaderStageCreateInfo],
	pub raster: &'a vk::PipelineRasterizationStateCreateInfo,
	pub depth: &'a vk::PipelineDepthStencilStateCreateInfo,
	pub multisample: &'a vk::PipelineMultisampleStateCreateInfo,
	pub blend: &'a vk::PipelineColorBlendStateCreateInfo,
	pub dynamic: &'a [vk::DynamicState],
	pub layout: vk::PipelineLayout,
	pub color_attachments: &'a [vk::Format],
	pub depth_attachment: vk::Format,
	pub stencil_attachment: vk::Format,
}

impl Default for GraphicsPipelineDesc<'_> {
	fn default() -> Self {
		const BLEND: vk::PipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo {
			s_type: vk::PipelineColorBlendStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineColorBlendStateCreateFlags::empty(),
			logic_op_enable: 0,
			logic_op: vk::LogicOp::NO_OP,
			attachment_count: 0,
			p_attachments: std::ptr::null(),
			blend_constants: [0.0, 0.0, 0.0, 0.0],
		};

		const RASTER: vk::PipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo {
			s_type: vk::PipelineRasterizationStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineRasterizationStateCreateFlags::empty(),
			depth_clamp_enable: 0,
			rasterizer_discard_enable: 0,
			polygon_mode: vk::PolygonMode::FILL,
			cull_mode: vk::CullModeFlags::BACK,
			front_face: vk::FrontFace::COUNTER_CLOCKWISE,
			depth_bias_enable: 0,
			depth_bias_constant_factor: 0.0,
			depth_bias_clamp: 0.0,
			depth_bias_slope_factor: 0.0,
			line_width: 1.0,
		};

		const STENCIL: vk::StencilOpState = vk::StencilOpState {
			fail_op: vk::StencilOp::REPLACE,
			pass_op: vk::StencilOp::REPLACE,
			depth_fail_op: vk::StencilOp::REPLACE,
			compare_op: vk::CompareOp::NEVER,
			compare_mask: 0,
			write_mask: 0,
			reference: 0,
		};

		const DEPTH: vk::PipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo {
			s_type: vk::PipelineDepthStencilStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
			depth_test_enable: 0,
			depth_write_enable: 0,
			depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
			depth_bounds_test_enable: 0,
			stencil_test_enable: 0,
			front: STENCIL,
			back: STENCIL,
			min_depth_bounds: 0.0,
			max_depth_bounds: 1.0,
		};

		const MULTISAMPLE: vk::PipelineMultisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo {
			s_type: vk::PipelineMultisampleStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineMultisampleStateCreateFlags::empty(),
			rasterization_samples: vk::SampleCountFlags::TYPE_1,
			sample_shading_enable: 0,
			min_sample_shading: 0.0,
			p_sample_mask: std::ptr::null(),
			alpha_to_coverage_enable: 0,
			alpha_to_one_enable: 0,
		};

		Self {
			layout: vk::PipelineLayout::null(),
			shaders: &[],
			color_attachments: &[],
			blend: &BLEND,
			// Values that can be defaulted below.
			depth_attachment: vk::Format::UNDEFINED,
			stencil_attachment: vk::Format::UNDEFINED,
			raster: &RASTER,
			depth: &DEPTH,
			multisample: &MULTISAMPLE,
			dynamic: &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
		}
	}
}

impl RenderCore {
	pub fn graphics_pipeline(&self, device: &CoreDevice, desc: &GraphicsPipelineDesc) -> Result<vk::Pipeline> {
		unsafe {
			let pipeline = device
				.device()
				.create_graphics_pipelines(
					self.cache.cache(),
					&[vk::GraphicsPipelineCreateInfo::builder()
						.stages(desc.shaders)
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
						.rasterization_state(desc.raster)
						.depth_stencil_state(desc.depth)
						.multisample_state(desc.multisample)
						.color_blend_state(desc.blend)
						.dynamic_state(&vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(desc.dynamic))
						.layout(desc.layout)
						.push_next(
							&mut vk::PipelineRenderingCreateInfo::builder()
								.color_attachment_formats(desc.color_attachments)
								.depth_attachment_format(desc.depth_attachment)
								.stencil_attachment_format(desc.stencil_attachment),
						)
						.build()],
					None,
				)
				.map_err(|x| x.1)?[0];

			Ok(pipeline)
		}
	}

	pub fn compute_pipeline(
		&self, device: &CoreDevice, layout: vk::PipelineLayout, shader: vk::PipelineShaderStageCreateInfoBuilder,
	) -> Result<vk::Pipeline> {
		unsafe {
			let pipeline = device
				.device()
				.create_compute_pipelines(
					self.cache.cache(),
					&[vk::ComputePipelineCreateInfo::builder()
						.layout(layout)
						.stage(shader.build())
						.build()],
					None,
				)
				.map_err(|x| x.1)?[0];

			Ok(pipeline)
		}
	}
}
