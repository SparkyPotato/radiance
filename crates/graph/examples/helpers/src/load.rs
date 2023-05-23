use std::io::Cursor;

pub use image::ImageFormat;
use radiance_graph::{
	ash::vk::{
		AccessFlags,
		BufferImageCopy,
		BufferUsageFlags,
		CommandBufferAllocateInfo,
		CommandBufferBeginInfo,
		CommandBufferLevel,
		CommandBufferSubmitInfo,
		CommandBufferUsageFlags,
		CommandPoolCreateFlags,
		CommandPoolCreateInfo,
		DependencyFlags,
		Extent3D,
		FenceCreateInfo,
		Format,
		ImageAspectFlags,
		ImageCreateFlags,
		ImageLayout,
		ImageMemoryBarrier,
		ImageSubresourceLayers,
		ImageSubresourceRange,
		ImageUsageFlags,
		ImageViewType,
		PipelineStageFlags,
		SampleCountFlags,
		SubmitInfo2,
	},
	device::Device,
	resource::{BufferDesc, Image, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource, UploadBuffer},
};

pub fn image(device: &Device, bytes: &[u8], format: ImageFormat) -> (Image, ImageView) {
	let image = image::load(Cursor::new(bytes), format).unwrap().to_rgba8();
	let buf = UploadBuffer::create(
		device,
		BufferDesc {
			size: image.len() as _,
			usage: BufferUsageFlags::TRANSFER_SRC,
		},
	)
	.unwrap();
	unsafe {
		buf.handle().data.as_mut().copy_from_slice(image.as_raw());
	}
	let size = Extent3D::builder()
		.width(image.width())
		.height(image.height())
		.depth(1)
		.build();
	let image = Image::create(
		device,
		ImageDesc {
			flags: ImageCreateFlags::empty(),
			format: Format::R8G8B8A8_SRGB,
			size,
			levels: 1,
			layers: 1,
			samples: SampleCountFlags::TYPE_1,
			usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
		},
	)
	.unwrap();
	let view = ImageView::create(
		device,
		ImageViewDesc {
			image: image.handle(),
			view_type: ImageViewType::TYPE_2D,
			format: Format::R8G8B8A8_SRGB,
			usage: ImageViewUsage::Sampled,
			aspect: ImageAspectFlags::COLOR,
		},
	)
	.unwrap();

	// Very inefficient - we're creating a command pool and submitting a command buffer for every image.
	// You should write your own, optimized resource loading code.
	unsafe {
		let pool = device
			.device()
			.create_command_pool(
				&CommandPoolCreateInfo::builder()
					.queue_family_index(*device.queue_families().graphics())
					.flags(CommandPoolCreateFlags::TRANSIENT),
				None,
			)
			.unwrap();
		let cmd_buf = device
			.device()
			.allocate_command_buffers(
				&CommandBufferAllocateInfo::builder()
					.command_pool(pool)
					.level(CommandBufferLevel::PRIMARY)
					.command_buffer_count(1),
			)
			.unwrap()[0];

		device
			.device()
			.begin_command_buffer(
				cmd_buf,
				&CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
			.unwrap();
		let range = ImageSubresourceRange::builder()
			.aspect_mask(ImageAspectFlags::COLOR)
			.base_array_layer(0)
			.layer_count(1)
			.base_mip_level(0)
			.level_count(1)
			.build();
		device.device().cmd_pipeline_barrier(
			cmd_buf,
			PipelineStageFlags::NONE,
			PipelineStageFlags::TRANSFER,
			DependencyFlags::empty(),
			&[],
			&[],
			&[ImageMemoryBarrier::builder()
				.image(image.handle())
				.old_layout(ImageLayout::UNDEFINED)
				.new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
				.src_access_mask(AccessFlags::NONE)
				.dst_access_mask(AccessFlags::TRANSFER_WRITE)
				.subresource_range(range)
				.build()],
		);
		device.device().cmd_copy_buffer_to_image(
			cmd_buf,
			buf.handle().buffer,
			image.handle(),
			ImageLayout::TRANSFER_DST_OPTIMAL,
			&[BufferImageCopy::builder()
				.image_subresource(
					ImageSubresourceLayers::builder()
						.base_array_layer(0)
						.layer_count(1)
						.aspect_mask(ImageAspectFlags::COLOR)
						.mip_level(0)
						.build(),
				)
				.image_extent(size)
				.build()],
		);
		device.device().cmd_pipeline_barrier(
			cmd_buf,
			PipelineStageFlags::TRANSFER,
			PipelineStageFlags::FRAGMENT_SHADER,
			DependencyFlags::empty(),
			&[],
			&[],
			&[ImageMemoryBarrier::builder()
				.image(image.handle())
				.old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				.src_access_mask(AccessFlags::TRANSFER_WRITE)
				.dst_access_mask(AccessFlags::SHADER_READ)
				.subresource_range(range)
				.build()],
		);
		device.device().end_command_buffer(cmd_buf).unwrap();

		let fence = device.device().create_fence(&FenceCreateInfo::builder(), None).unwrap();

		device
			.submit_graphics(
				&[SubmitInfo2::builder()
					.command_buffer_infos(&[CommandBufferSubmitInfo::builder().command_buffer(cmd_buf).build()])
					.build()],
				fence,
			)
			.unwrap();
		device.device().wait_for_fences(&[fence], true, u64::MAX).unwrap();

		buf.destroy(device);
		device.device().destroy_command_pool(pool, None);
		device.device().destroy_fence(fence, None);
	}

	(image, view)
}
