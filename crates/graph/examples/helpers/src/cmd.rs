use radiance_graph::{
	ash::vk::{
		AttachmentLoadOp,
		AttachmentStoreOp,
		ClearColorValue,
		ClearValue,
		CommandBuffer,
		Extent2D,
		ImageLayout,
		Rect2D,
		RenderingAttachmentInfo,
		RenderingInfo,
		Viewport,
	},
	device::Device,
	resource::ImageView,
};
use vek::Vec2;

use crate::misc::simple_rect;

pub fn set_viewport_and_scissor(device: &Device, buf: CommandBuffer, size: Vec2<u32>) {
	unsafe {
		device.device().cmd_set_viewport(
			buf,
			0,
			&[Viewport {
				x: 0.0,
				y: 0.0,
				width: size.x as f32,
				height: size.y as f32,
				min_depth: 0.0,
				max_depth: 1.0,
			}],
		);
		device.device().cmd_set_scissor(buf, 0, &[simple_rect(size)]);
	}
}

pub fn start_rendering_swapchain(device: &Device, buf: CommandBuffer, view: ImageView, size: Vec2<u32>) {
	unsafe {
		device.device().cmd_begin_rendering(
			buf,
			&RenderingInfo::builder()
				.render_area(
					Rect2D::builder()
						.extent(Extent2D::builder().width(size.x).height(size.y).build())
						.build(),
				)
				.layer_count(1)
				.color_attachments(&[RenderingAttachmentInfo::builder()
					.image_view(view.view)
					.image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(AttachmentLoadOp::CLEAR)
					.clear_value(ClearValue {
						color: ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(AttachmentStoreOp::STORE)
					.build()]),
		);
		set_viewport_and_scissor(device, buf, size);
	}
}
