use std::io::{self, Write};

use ash::vk;
use bincode::{Decode, Encode};
// use nvtt_rs::{CompressionOptions, Container, Context, Format, InputFormat, OutputOptions, Surface,
// CUDA_SUPPORTED};
use rad_core::{
	asset::{AssetView, BincodeAsset, CookedAsset, Uuid},
	uuid,
	Engine,
};
use rad_graph::{
	cmd::CommandPool,
	device::{descriptor::ImageId, Device, QueueWait, Transfer},
	resource::{
		Buffer,
		BufferDesc,
		BufferType,
		Image,
		ImageDesc,
		ImageView,
		ImageViewDesc,
		ImageViewUsage,
		Resource,
		Subresource,
	},
	sync::{get_image_barrier, ImageBarrier, UsageType},
};
use tracing::trace_span;
use vek::Vec3;

#[derive(Encode, Decode)]
pub struct ImageAsset {
	#[bincode(with_serde)]
	pub size: Vec3<u32>,
	/// This is a  `vk::Format` but that's not serializable so...
	pub format: i32,
	pub data: Vec<u8>,
}

impl BincodeAsset for ImageAsset {
	const UUID: Uuid = uuid!("e68fac6b-41d0-48c5-a5ff-3e6cfe9b53f0");
}

impl CookedAsset for ImageAsset {
	type Base = ImageAsset;

	fn cook(base: &Self::Base) -> Self {
		Self {
			size: base.size,
			format: base.format,
			data: base.data.clone(),
		}

		// TODO: bad, swizzle to support more formats.
		// let in_fmt = vk::Format::from_raw(base.format);
		// let (in_fmt, out_fmt, out_vk_fmt) =
		// 	if in_fmt == vk::Format::B8G8R8A8_UNORM || in_fmt == vk::Format::B8G8R8A8_SRGB {
		// 		(
		// 			InputFormat::Bgra8Ub {
		// 				data: &base.data,
		// 				unsigned_to_signed: false,
		// 			},
		// 			Format::Bc7,
		// 			if in_fmt == vk::Format::B8G8R8A8_SRGB {
		// 				vk::Format::BC7_SRGB_BLOCK
		// 			} else {
		// 				vk::Format::BC7_UNORM_BLOCK
		// 			},
		// 		)
		// 	} else if in_fmt == vk::Format::B8G8R8A8_SNORM {
		// 		(
		// 			InputFormat::Bgra8Sb(&base.data),
		// 			Format::Bc7,
		// 			vk::Format::BC7_UNORM_BLOCK,
		// 		)
		// 	} else if in_fmt == vk::Format::R16G16B16A16_SFLOAT {
		// 		(
		// 			InputFormat::Rgba16f(&base.data),
		// 			Format::Bc6S,
		// 			vk::Format::BC6H_SFLOAT_BLOCK,
		// 		)
		// 	} else if in_fmt == vk::Format::R32G32B32A32_SFLOAT {
		// 		(
		// 			InputFormat::Rgba32f(&base.data),
		// 			Format::Bc6S,
		// 			vk::Format::BC6H_SFLOAT_BLOCK,
		// 		)
		// 	} else if in_fmt == vk::Format::R32_SFLOAT {
		// 		(
		// 			InputFormat::R32f(&base.data),
		// 			Format::Bc6S,
		// 			vk::Format::BC6H_SFLOAT_BLOCK,
		// 		)
		// 	} else {
		// 		panic!("unsupported format")
		// 	};
		// let image = Surface::image(in_fmt, base.size.x, base.size.y, base.size.z).expect("invalid data");
		//
		// let mut context = Context::new();
		// if *CUDA_SUPPORTED {
		// 	context.set_cuda_acceleration(true);
		// }
		//
		// let mut opts = CompressionOptions::new();
		// opts.set_format(out_fmt);
		//
		// let mut out_opts = OutputOptions::new();
		// out_opts.set_srgb_flag(true);
		// out_opts.set_output_header(false);
		// out_opts.set_container(Container::Dds10);
		//
		// Self {
		// 	size: base.size,
		// 	format: out_vk_fmt.as_raw(),
		// 	data: context.compress(&image, &opts, &out_opts).unwrap(),
		// }
	}
}

pub struct ImageAssetView {
	image: Image,
	view: ImageView,
}

impl ImageAssetView {
	pub fn image(&self) -> &Image { &self.image }

	pub fn view(&self) -> &ImageView { &self.view }

	pub fn image_id(&self) -> ImageId { self.view.id.unwrap() }

	pub fn new(name: &str, data: ImageAsset) -> Result<Self, std::io::Error> {
		let s = trace_span!("load image", name = name);
		let _e = s.enter();

		let device: &Device = Engine::get().global();
		let size = vk::Extent3D {
			width: data.size.x,
			height: data.size.y,
			depth: data.size.z,
		};
		let format = vk::Format::from_raw(data.format);
		let image = Image::create(
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
		let staging = Buffer::create(
			device,
			BufferDesc {
				name: &format!("{name} staging buffer"),
				size: data.data.len() as _,
				ty: BufferType::Staging,
			},
		)?;
		unsafe {
			let mut pool = CommandPool::new(device, device.queue_families().into::<Transfer>())?;
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
			let sync = device.submit::<Transfer>(QueueWait::default(), &[cmd], &[])?;
			sync.wait(device)?;
			pool.destroy(device);
			staging.destroy(device);
		}

		let view = ImageView::create(
			device,
			ImageViewDesc {
				name: &format!("{name} view"),
				image: image.handle(),
				view_type: if size.depth == 1 {
					vk::ImageViewType::TYPE_2D
				} else {
					vk::ImageViewType::TYPE_3D
				},
				format,
				usage: ImageViewUsage::Sampled,
				size,
				subresource: Subresource::default(),
			},
		)?;

		Ok(Self { image, view })
	}
}

impl AssetView for ImageAssetView {
	type Base = ImageAsset;
	type Ctx = ();

	fn load(_: &'static Self::Ctx, base: Self::Base) -> Result<Self, io::Error> {
		// TODO: fix
		Self::new("image asset", base)
	}
}
