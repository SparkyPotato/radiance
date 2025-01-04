use std::io::{self, Write};

use ash::vk;
use bincode::{config::standard, Decode, Encode};
use nvtt_rs::{CompressionOptions, Container, Context, Format, InputFormat, OutputOptions, Surface, CUDA_SUPPORTED};
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

#[derive(Encode, Decode)]
struct ImageData {
	width: u32,
	height: u32,
	srgb: bool,
	data: Vec<u8>,
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
		view.seek_begin()?;
		let data: ImageData = bincode::decode_from_std_read(&mut view.read_section()?, standard())
			.map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
		let name = view.name();

		let device: &Device = Engine::get().global();
		let size = vk::Extent3D {
			width: data.width,
			height: data.height,
			depth: 1,
		};
		let format = if data.srgb {
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
				size: data.data.len() as _,
				readback: false,
			},
		)?;
		unsafe {
			let mut pool = CommandPool::new(device, device.queue_families().transfer)?;
			let cmd = pool.next(device)?;
			staging.data().as_mut().write_all(&data.data)?;
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

		let image = Surface::image(
			InputFormat::Bgra8Ub {
				data: data.data,
				unsigned_to_signed: false,
			},
			data.width,
			data.height,
			1,
		)
		.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{:?}", e)))?;

		let mut context = Context::new();
		if *CUDA_SUPPORTED {
			context.set_cuda_acceleration(true);
		}

		let mut opts = CompressionOptions::new();
		opts.set_format(Format::Bc7);

		let mut out_opts = OutputOptions::new();
		out_opts.set_srgb_flag(true);
		out_opts.set_output_header(false);
		out_opts.set_container(Container::Dds10);

		let write = context.compress(&image, &opts, &out_opts).unwrap();
		into.clear()?;
		bincode::encode_into_std_write(
			ImageData {
				width: data.width,
				height: data.height,
				srgb: data.is_srgb,
				data: write,
			},
			&mut into.new_section()?,
			standard(),
		)
		.map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

		Ok(())
	}
}
