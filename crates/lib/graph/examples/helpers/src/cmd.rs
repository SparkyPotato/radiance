use radiance_graph::{ash::vk, device::Device, resource::ImageView};

pub fn set_viewport_and_scissor(device: &Device, buf: vk::CommandBuffer, size: vk::Extent2D) {
	unsafe {
		device.device().cmd_set_viewport(
			buf,
			0,
			&[vk::Viewport {
				x: 0.0,
				y: 0.0,
				width: size.width as f32,
				height: size.height as f32,
				min_depth: 0.0,
				max_depth: 1.0,
			}],
		);
		device
			.device()
			.cmd_set_scissor(buf, 0, &[vk::Rect2D::builder().extent(size).build()]);
	}
}

pub fn start_rendering_swapchain(device: &Device, buf: vk::CommandBuffer, view: ImageView, size: vk::Extent2D) {
	unsafe {
		device.device().cmd_begin_rendering(
			buf,
			&vk::RenderingInfo::builder()
				.render_area(vk::Rect2D::builder().extent(size).build())
				.layer_count(1)
				.color_attachments(&[vk::RenderingAttachmentInfo::builder()
					.image_view(view.view)
					.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.clear_value(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(vk::AttachmentStoreOp::STORE)
					.build()]),
		);
		set_viewport_and_scissor(device, buf, size);
	}
}
