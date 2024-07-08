use ash::vk;
use crossbeam_channel::Sender;
use radiance_asset::{image::Format, Asset, AssetSource};
use radiance_graph::{
	graph::Resource,
	resource::{Image as GImage, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource as _, Subresource},
};
use uuid::Uuid;

use crate::asset::{
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
		queue.send(Resource::Image(self.image).into()).unwrap();
		queue.send(Resource::ImageView(self.view).into()).unwrap();
	}
}

impl<S: AssetSource> Loader<'_, S> {
	pub fn load_image(&mut self, uuid: Uuid, srgb: bool) -> LResult<Image, S> {
		match AssetRuntime::get_cache(&mut self.runtime.images, uuid) {
			Some(x) => Ok(x),
			None => {
				let i = self.load_image_from_disk(uuid, srgb)?;
				self.runtime.images.insert(uuid, i.downgrade());
				Ok(i)
			},
		}
	}

	fn load_image_from_disk(&mut self, image: Uuid, srgb: bool) -> LResult<Image, S> {
		let Asset::Image(i) = self.sys.load(image)? else {
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
		let name = self.sys.human_name(image).unwrap_or("unnamed image".to_string());
		let size = vk::Extent3D::default().width(i.width).height(i.height).depth(1);
		let img = GImage::create(
			self.device,
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
			self.device,
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

		Ok(RRef::new(Image { image: img, view }, self.runtime.deleter.clone()))
	}
}
