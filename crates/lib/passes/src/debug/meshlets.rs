use ash::vk;
use radiance_core::CoreFrame;
use radiance_graph::{
	graph::{ImageUsage, ImageUsageType, ReadId, Shader},
	resource::ImageView,
};

pub struct DebugMeshlets {}

impl DebugMeshlets {
	pub fn new() -> Self { Self {} }

	pub fn run<'pass>(
		&'pass self, frame: &mut CoreFrame<'pass, '_>, visbuffer: ReadId<ImageView>,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("debug meshlets");
		pass.input(
			visbuffer,
			ImageUsage {
				format: vk::Format::R32_UINT,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);
		let (ret, output) = pass.output(
			visbuffer,
			ImageUsage {
				format: vk::Format::R8G8B8A8_SRGB,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);

		pass.build(|_| {});

		ret
	}
}
