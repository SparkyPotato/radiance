use ash::vk;

pub fn reverse_depth() -> vk::PipelineDepthStencilStateCreateInfo<'static> {
	vk::PipelineDepthStencilStateCreateInfo::default()
		.depth_test_enable(true)
		.depth_write_enable(true)
		.depth_compare_op(vk::CompareOp::GREATER)
}

pub fn no_cull() -> vk::PipelineRasterizationStateCreateInfo<'static> {
	vk::PipelineRasterizationStateCreateInfo::default()
		.polygon_mode(vk::PolygonMode::FILL)
		.front_face(vk::FrontFace::CLOCKWISE)
		.cull_mode(vk::CullModeFlags::NONE)
		.line_width(1.0)
}

pub fn simple_blend(states: &[vk::PipelineColorBlendAttachmentState]) -> vk::PipelineColorBlendStateCreateInfo {
	vk::PipelineColorBlendStateCreateInfo::default().attachments(states)
}

pub fn no_blend() -> vk::PipelineColorBlendAttachmentState {
	vk::PipelineColorBlendAttachmentState::default()
		.color_write_mask(
			vk::ColorComponentFlags::R
				| vk::ColorComponentFlags::G
				| vk::ColorComponentFlags::B
				| vk::ColorComponentFlags::A,
		)
		.blend_enable(false)
}

pub fn default_blend() -> vk::PipelineColorBlendAttachmentState {
	vk::PipelineColorBlendAttachmentState::default()
		.color_write_mask(
			vk::ColorComponentFlags::R
				| vk::ColorComponentFlags::G
				| vk::ColorComponentFlags::B
				| vk::ColorComponentFlags::A,
		)
		.blend_enable(true)
		.src_color_blend_factor(vk::BlendFactor::ONE)
		.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
		.color_blend_op(vk::BlendOp::ADD)
		.src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
		.dst_alpha_blend_factor(vk::BlendFactor::ONE)
		.alpha_blend_op(vk::BlendOp::ADD)
}
