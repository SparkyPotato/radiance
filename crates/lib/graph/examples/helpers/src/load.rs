use std::io::Cursor;

pub use image::ImageFormat;
use radiance_graph::{
	ash::vk,
	cmd::CommandPool,
	device::Device,
	resource::{BufferDesc, Image, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource, UploadBuffer},
};

pub fn image(device: &Device, bytes: &[u8], format: ImageFormat) -> (Image, ImageView) {
	let image = image::load(Cursor::new(bytes), format).unwrap().to_rgba8();
	let buf = UploadBuffer::create(
		device,
		BufferDesc {
			size: image.len() as _,
			usage: vk::BufferUsageFlags::TRANSFER_SRC,
		},
	)
	.unwrap();
	unsafe {
		let raw = image.as_raw();
		buf.handle().data.as_mut()[..raw.len()].copy_from_slice(raw);
	}
	let size = vk::Extent3D::builder()
		.width(image.width())
		.height(image.height())
		.depth(1)
		.build();
	let image = Image::create(
		device,
		ImageDesc {
			flags: vk::ImageCreateFlags::empty(),
			format: vk::Format::R8G8B8A8_SRGB,
			size,
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
			usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
		},
	)
	.unwrap();
	let view = ImageView::create(
		device,
		ImageViewDesc {
			size,
			image: image.handle(),
			view_type: vk::ImageViewType::TYPE_2D,
			format: vk::Format::R8G8B8A8_SRGB,
			usage: ImageViewUsage::Sampled,
			aspect: vk::ImageAspectFlags::COLOR,
		},
	)
	.unwrap();

	// Very inefficient - we're creating a command pool and submitting a command buffer for every image.
	// You should write your own, optimized resource loading code.
	unsafe {
		let mut pool = CommandPool::new(device, *device.queue_families().graphics()).unwrap();
		let cmd_buf = pool.next(device).unwrap();

		device
			.device()
			.begin_command_buffer(
				cmd_buf,
				&vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
			.unwrap();
		let range = vk::ImageSubresourceRange::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.base_array_layer(0)
			.layer_count(1)
			.base_mip_level(0)
			.level_count(1)
			.build();
		device.device().cmd_pipeline_barrier2(
			cmd_buf,
			&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
				.image(image.handle())
				.old_layout(vk::ImageLayout::UNDEFINED)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.src_access_mask(vk::AccessFlags2::NONE)
				.dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
				.src_stage_mask(vk::PipelineStageFlags2::NONE)
				.dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
				.subresource_range(range)
				.build()]),
		);
		device.device().cmd_copy_buffer_to_image(
			cmd_buf,
			buf.handle().buffer,
			image.handle(),
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			&[vk::BufferImageCopy::builder()
				.image_subresource(
					vk::ImageSubresourceLayers::builder()
						.base_array_layer(0)
						.layer_count(1)
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.mip_level(0)
						.build(),
				)
				.image_extent(size)
				.build()],
		);
		device.device().cmd_pipeline_barrier2(
			cmd_buf,
			&vk::DependencyInfo::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2::builder()
				.image(image.handle())
				.old_layout(vk::ImageLayout::UNDEFINED)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
				.dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
				.src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
				.dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
				.subresource_range(range)
				.build()]),
		);
		device.device().end_command_buffer(cmd_buf).unwrap();

		let fence = device
			.device()
			.create_fence(&vk::FenceCreateInfo::builder(), None)
			.unwrap();

		device
			.submit_graphics(
				&[vk::SubmitInfo2::builder()
					.command_buffer_infos(&[vk::CommandBufferSubmitInfo::builder().command_buffer(cmd_buf).build()])
					.build()],
				fence,
			)
			.unwrap();
		device.device().wait_for_fences(&[fence], true, u64::MAX).unwrap();

		buf.destroy(device);
		pool.destroy(device);
		device.device().destroy_fence(fence, None);
	}

	(image, view)
}
