use ash::vk;
use radiance_graph::{device::Device, Result};

pub struct PipelineCache {
	inner: vk::PipelineCache,
}

impl PipelineCache {
	pub fn new(device: &Device) -> Result<Self> {
		let cache = unsafe {
			device
				.device()
				.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)?
		};
		Ok(Self { inner: cache })
	}

	pub fn cache(&self) -> vk::PipelineCache { self.inner }

	pub fn dump(&self, device: &Device) -> Result<Vec<u8>> {
		unsafe {
			let data = device.device().get_pipeline_cache_data(self.inner)?;
			Ok(data)
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_pipeline_cache(self.inner, None);
		}
	}
}

pub fn reverse_depth() -> vk::PipelineDepthStencilStateCreateInfoBuilder<'static> {
	vk::PipelineDepthStencilStateCreateInfo::builder()
		.depth_test_enable(true)
		.depth_write_enable(true)
		.depth_compare_op(vk::CompareOp::GREATER)
}

pub fn no_cull() -> vk::PipelineRasterizationStateCreateInfoBuilder<'static> {
	vk::PipelineRasterizationStateCreateInfo::builder()
		.polygon_mode(vk::PolygonMode::FILL)
		.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.cull_mode(vk::CullModeFlags::NONE)
		.line_width(1.0)
}

pub fn simple_blend(states: &[vk::PipelineColorBlendAttachmentState]) -> vk::PipelineColorBlendStateCreateInfoBuilder {
	vk::PipelineColorBlendStateCreateInfo::builder().attachments(states)
}

pub fn no_blend() -> vk::PipelineColorBlendAttachmentState {
	vk::PipelineColorBlendAttachmentState::builder()
		.color_write_mask(
			vk::ColorComponentFlags::R
				| vk::ColorComponentFlags::G
				| vk::ColorComponentFlags::B
				| vk::ColorComponentFlags::A,
		)
		.blend_enable(false)
		.build()
}

pub fn default_blend() -> vk::PipelineColorBlendAttachmentState {
	vk::PipelineColorBlendAttachmentState::builder()
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
		.build()
}
