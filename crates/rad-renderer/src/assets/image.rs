use std::io::{self, Write};

use ash::vk;
use basis_universal::{
	encoder_init,
	transcoder_init,
	BasisTextureFormat,
	ColorSpace,
	Compressor,
	CompressorParams,
	DecodeFlags,
	TranscodeParameters,
	Transcoder,
	TranscoderTextureFormat,
	UserData,
};
use rad_core::{
	asset::{Asset, AssetView, Uuid},
	uuid,
	Engine,
};
use rad_graph::{
	cmd::CommandPool,
	device::{Device, QueueWait, Transfer},
	resource::{self, Buffer, BufferDesc, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource, Subresource},
	sync::{get_image_barrier, ImageBarrier, UsageType},
};
use tracing::trace_span;

pub struct Image {
	image: resource::Image,
	view: ImageView,
}

impl Asset for Image {
	fn uuid() -> Uuid
	where
		Self: Sized,
	{
		uuid!("e68fac6b-41d0-48c5-a5ff-3e6cfe9b53f0")
	}

	fn unloaded() -> Self
	where
		Self: Sized,
	{
		Self {
			image: resource::Image::default(),
			view: ImageView::default(),
		}
	}

	fn load(mut view: Box<dyn AssetView>) -> Result<Self, std::io::Error>
	where
		Self: Sized,
	{
		transcoder_init();
		view.seek_begin()?;
		let mut read = view.read_section()?;
		let mut data = Vec::new();
		read.read_to_end(&mut data)?;
		drop(read);
		let name = view.name();

		let mut trans = Transcoder::new();
		trans
			.prepare_transcoding(&data)
			.map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "failed to decode image"))?;
		let info = trans
			.image_level_info(&data, 0, 0)
			.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "image missing"))?;
		let is_srgb = trans
			.user_data(&data)
			.map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "failed to decode image"))?
			.userdata0
			!= 0;
		let data = trans
			.transcode_image_level(
				&data,
				TranscoderTextureFormat::BC7_RGBA,
				TranscodeParameters {
					image_index: 0,
					level_index: 0,
					decode_flags: Some(DecodeFlags::HIGH_QUALITY),
					output_rows_in_pixels: None,
					output_row_pitch_in_blocks_or_pixels: None,
				},
			)
			.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{:?}", e)))?;
		trans.end_transcoding();

		let device: &Device = Engine::get().global();
		let size = vk::Extent3D {
			width: info.m_width,
			height: info.m_height,
			depth: 1,
		};
		let format = if is_srgb {
			vk::Format::BC7_SRGB_BLOCK
		} else {
			vk::Format::BC7_UNORM_BLOCK
		};
		let image = resource::Image::create(
			device,
			ImageDesc {
				name,
				size,
				format,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				flags: vk::ImageCreateFlags::empty(),
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			},
		)?;
		// TODO: not device local
		let staging = Buffer::create(
			device,
			BufferDesc {
				name: &format!("{name} staging buffer"),
				size: data.len() as _,
				readback: false,
			},
		)?;
		unsafe {
			let mut pool = CommandPool::new(device, device.queue_families().transfer)?;
			let cmd = pool.next(device)?;
			staging.data().as_mut().write_all(&data)?;
			device
				.device()
				.begin_command_buffer(
					cmd,
					&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)
				.unwrap();
			device.device().cmd_pipeline_barrier2(
				cmd,
				&vk::DependencyInfo::default().image_memory_barriers(&[get_image_barrier(&ImageBarrier {
					previous_usages: &[],
					next_usages: &[UsageType::TransferWrite],
					discard_contents: true,
					image: image.handle(),
					range: vk::ImageSubresourceRange::default()
						.base_array_layer(0)
						.layer_count(1)
						.base_mip_level(0)
						.level_count(1)
						.aspect_mask(vk::ImageAspectFlags::COLOR),
				})]),
			);
			device.device().cmd_copy_buffer_to_image2(
				cmd,
				&vk::CopyBufferToImageInfo2::default()
					.src_buffer(staging.inner())
					.dst_image(image.handle())
					.dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.regions(&[vk::BufferImageCopy2::default()
						.buffer_offset(0)
						.buffer_row_length(0)
						.buffer_image_height(0)
						.image_subresource(
							vk::ImageSubresourceLayers::default()
								.base_array_layer(0)
								.layer_count(1)
								.mip_level(0)
								.aspect_mask(vk::ImageAspectFlags::COLOR),
						)
						.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
						.image_extent(size)]),
			);
			device.device().cmd_pipeline_barrier2(
				cmd,
				&vk::DependencyInfo::default().image_memory_barriers(&[get_image_barrier(&ImageBarrier {
					previous_usages: &[UsageType::TransferWrite],
					next_usages: &[UsageType::OverrideLayout(vk::ImageLayout::READ_ONLY_OPTIMAL)],
					discard_contents: false,
					image: image.handle(),
					range: vk::ImageSubresourceRange::default()
						.base_array_layer(0)
						.layer_count(1)
						.base_mip_level(0)
						.level_count(1)
						.aspect_mask(vk::ImageAspectFlags::COLOR),
				})]),
			);
			device.device().end_command_buffer(cmd).unwrap();
			let sync = device.submit::<Transfer>(QueueWait::default(), &[cmd], &[], vk::Fence::null())?;
			sync.wait(device)?;
			pool.destroy(device);
			staging.destroy(device);
		}

		let view = ImageView::create(
			device,
			ImageViewDesc {
				name: &format!("{name} view"),
				image: image.handle(),
				view_type: vk::ImageViewType::TYPE_2D,
				format,
				usage: ImageViewUsage::Sampled,
				size,
				subresource: Subresource::default(),
			},
		)?;

		Ok(Self { image, view })
	}

	fn save(&self, _: &mut dyn AssetView) -> Result<(), std::io::Error> {
		Err(io::Error::new(io::ErrorKind::Unsupported, "images cannot be edited"))
	}
}

pub struct ImportImage<'a> {
	pub data: &'a [u8],
	pub width: u32,
	pub height: u32,
	pub is_normal_map: bool,
	pub is_srgb: bool,
}

impl Image {
	pub fn image(&self) -> &resource::Image { &self.image }

	pub fn view(&self) -> &ImageView { &self.view }

	pub fn import(name: &str, data: ImportImage, mut into: Box<dyn AssetView>) -> Result<(), io::Error> {
		let s = trace_span!("import image", name = name);
		let _e = s.enter();

		encoder_init();
		let mut comp = Compressor::new(std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) as _);
		let mut params = CompressorParams::new();
		params.source_image_mut(0).init(
			data.data,
			data.width,
			data.height,
			(data.data.len() as u32 / (data.width * data.height)) as _,
		);
		params.set_basis_format(BasisTextureFormat::UASTC4x4);
		params.set_rdo_uastc(Some(1.0));
		params.set_color_space(if data.is_srgb {
			ColorSpace::Srgb
		} else {
			ColorSpace::Linear
		});
		if data.is_normal_map {
			params.tune_for_normal_maps();
		}
		params.set_userdata(UserData {
			userdata0: data.is_srgb as _,
			userdata1: 0,
		});
		unsafe {
			comp.init(&params);
			comp.process()
				.map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
		}

		into.clear()?;
		let mut write = into.new_section()?;
		write.write_all(comp.basis_file())?;

		Ok(())
	}
}
