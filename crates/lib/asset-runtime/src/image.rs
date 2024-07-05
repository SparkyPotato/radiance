use ash::vk;
use crossbeam_channel::Sender;
use radiance_asset::{image::Format, Asset, AssetSource};
use radiance_graph::{
	graph::Resource,
	resource::{Image as GImage, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource as _, Subresource},
};
use uuid::Uuid;

use crate::{
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	LoadError,
	Loader,
};

pub struct Image {
	pub image: GImage,
	pub view: ImageView,
}

impl RuntimeAsset for Image {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(DelRes::Resource(Resource::Image(self.image))).unwrap();
		queue.send(DelRes::Resource(Resource::ImageView(self.view))).unwrap();
	}
}

impl AssetRuntime {
	pub(crate) fn load_image_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, S>, image: Uuid, srgb: bool,
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
		let name = loader.sys.human_name(image).unwrap_or("unnamed image".to_string());
		let size = vk::Extent3D::builder().width(i.width).height(i.height).depth(1).build();
		let img = GImage::create(
			loader.device,
			ImageDesc {
				name: &name,
				flags: vk::ImageCreateFlags::empty(),
				format,
				size,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			},
		)
		.map_err(LoadError::Vulkan)?;
		let view = ImageView::create(
			loader.device,
			ImageViewDesc {
				name: &name,
				image: img.handle(),
				view_type: vk::ImageViewType::TYPE_2D,
				format,
				usage: ImageViewUsage::Sampled,
				subresource: Subresource::default(),
				size,
			},
		)
		.map_err(LoadError::Vulkan)?;
		// TODO: Write data

		Ok(RRef::new(Image { image: img, view }, loader.deleter.clone()))
	}
}
