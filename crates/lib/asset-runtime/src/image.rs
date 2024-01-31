use ash::vk;
use crossbeam_channel::Sender;
use radiance_asset::{image::Format, Asset, AssetSource};
use radiance_graph::{
	device::QueueType,
	resource::{Image as GImage, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource},
	sync::{ImageUsage, Shader},
};
use radiance_util::{
	deletion::IntoResource,
	staging::{ImageStage, StageError},
};
use uuid::Uuid;

use crate::{
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	Loader,
};

pub struct Image {
	pub image: GImage,
	pub view: ImageView,
}

impl RuntimeAsset for Image {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(self.image.into_resource().into()).unwrap();
		queue.send(self.view.into_resource().into()).unwrap();
	}
}

impl AssetRuntime {
	pub(crate) fn load_image_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, '_, '_, S>, image: Uuid, srgb: bool,
	) -> LResult<Image, S> {
		let Asset::Image(i) = loader.sys.load(image)? else {
			unreachable!("image asset is not image");
		};

		let format = match i.format {
			Format::R8 => {
				if srgb {
					vk::Format::R8_SRGB
				} else {
					vk::Format::R8_UNORM
				}
			},
			Format::R8G8 => {
				if srgb {
					vk::Format::R8G8_SRGB
				} else {
					vk::Format::R8G8_UNORM
				}
			},
			Format::R8G8B8A8 => {
				if srgb {
					vk::Format::R8G8B8A8_SRGB
				} else {
					vk::Format::R8G8B8A8_UNORM
				}
			},
			Format::R16 => vk::Format::R16_UNORM,
			Format::R16G16 => vk::Format::R16G16_UNORM,
			Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
			Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
			Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
			Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
		};
		let size = vk::Extent3D::builder().width(i.width).height(i.height).depth(1).build();
		let img = GImage::create(
			loader.device,
			ImageDesc {
				flags: vk::ImageCreateFlags::empty(),
				format,
				size,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			},
		)
		.map_err(StageError::Vulkan)?;
		let view = ImageView::create(
			loader.device,
			ImageViewDesc {
				image: img.handle(),
				view_type: vk::ImageViewType::TYPE_2D,
				format,
				usage: ImageViewUsage::Sampled,
				aspect: vk::ImageAspectFlags::COLOR,
				size,
			},
		)
		.map_err(StageError::Vulkan)?;
		loader
			.ctx
			.stage_image(
				&i.data,
				img.handle(),
				ImageStage {
					buffer_row_length: 0,
					buffer_image_height: 0,
					image_subresource: vk::ImageSubresourceLayers::builder()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.mip_level(0)
						.base_array_layer(0)
						.layer_count(1)
						.build(),
					image_offset: vk::Offset3D::default(),
					image_extent: size,
				},
				true,
				QueueType::Graphics,
				&[],
				&[ImageUsage::ShaderReadSampledImage(Shader::Any)],
			)
			.map_err(StageError::Vulkan)?;

		Ok(RRef::new(Image { image: img, view }, loader.deleter.clone()))
	}
}
