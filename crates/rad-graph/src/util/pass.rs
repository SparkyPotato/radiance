use ash::vk;
use bytemuck::{bytes_of, NoUninit};

use crate::{
	arena::IteratorAlloc,
	device::{ComputePipeline, GraphicsPipeline},
	graph::{PassContext, Res},
	resource::{BufferHandle, ImageView},
};

impl<'frame, 'graph> PassContext<'frame, 'graph> {
	pub fn bind_compute(&mut self, pipe: &ComputePipeline) { pipe.bind(self.device, self.buf); }

	pub fn push(&mut self, offset: usize, value: &impl NoUninit) {
		unsafe {
			self.device.device().cmd_push_constants(
				self.buf,
				self.device.layout(),
				vk::ShaderStageFlags::ALL,
				offset as _,
				bytes_of(value),
			);
		}
	}

	pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
		unsafe {
			self.device.device().cmd_dispatch(self.buf, x, y, z);
		}
	}

	pub fn dispatch_indirect(&mut self, buf: Res<BufferHandle>, offset: usize) {
		unsafe {
			let buf = self.get(buf).buffer;
			self.device.device().cmd_dispatch_indirect(self.buf, buf, offset as _);
		}
	}

	pub fn render_pass(
		&mut self, y_up: bool, attachments: &[Attachment], depth: Option<&Attachment>,
	) -> RenderPass<'_, 'frame, 'graph> {
		unsafe {
			let size = self
				.get(
					attachments
						.get(0)
						.or(depth)
						.expect("need atleast one attachment, use `empty_render_pass` otherwise")
						.image,
				)
				.size;
			let size = vk::Extent2D::default().width(size.width).height(size.height);
			let arena = self.arena;
			let attachments: Vec<_, _> = attachments.iter().map(|x| map_attachment(self, x)).collect_in(arena);
			let area = vk::Rect2D::default().extent(size);
			let info = vk::RenderingInfo::default()
				.render_area(area)
				.layer_count(1)
				.color_attachments(&attachments);

			match depth {
				Some(x) => self
					.device
					.device()
					.cmd_begin_rendering(self.buf, &info.depth_attachment(&map_attachment(self, x))),
				None => self.device.device().cmd_begin_rendering(self.buf, &info),
			}
			self.post_begin_render_pass(y_up, size);

			RenderPass { pass: self }
		}
	}

	pub fn empty_render_pass(&mut self, y_up: bool, size: vk::Extent2D) -> RenderPass<'_, 'frame, 'graph> {
		unsafe {
			self.device.device().cmd_begin_rendering(
				self.buf,
				&vk::RenderingInfo::default()
					.render_area(vk::Rect2D::default().extent(size))
					.layer_count(1),
			);
			self.post_begin_render_pass(y_up, size);
		}

		RenderPass { pass: self }
	}

	unsafe fn post_begin_render_pass(&self, y_up: bool, size: vk::Extent2D) {
		let width = size.width as f32;
		let height = size.height as f32;
		self.device.device().cmd_set_viewport(
			self.buf,
			0,
			&[if y_up {
				vk::Viewport {
					x: 0.0,
					y: height,
					width,
					height: -height,
					min_depth: 0.0,
					max_depth: 1.0,
				}
			} else {
				vk::Viewport {
					x: 0.0,
					y: 0.0,
					width,
					height,
					min_depth: 0.0,
					max_depth: 1.0,
				}
			}],
		);
		self.device
			.device()
			.cmd_set_scissor(self.buf, 0, &[vk::Rect2D::default().extent(size)]);
	}
}

fn map_attachment(pass: &mut PassContext, x: &Attachment) -> vk::RenderingAttachmentInfo<'static> {
	vk::RenderingAttachmentInfo::default()
		.image_view(pass.get(x.image).view)
		.image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
		.load_op(match x.load {
			Load::Load => vk::AttachmentLoadOp::LOAD,
			Load::Clear(_) => vk::AttachmentLoadOp::CLEAR,
			Load::DontCare => vk::AttachmentLoadOp::DONT_CARE,
		})
		.store_op(if x.store {
			vk::AttachmentStoreOp::STORE
		} else {
			vk::AttachmentStoreOp::DONT_CARE
		})
		.clear_value(match x.load {
			Load::Clear(x) => x,
			_ => vk::ClearValue::default(),
		})
}

#[derive(Copy, Clone)]
pub enum Load {
	Load,
	Clear(vk::ClearValue),
	DontCare,
}

pub struct Attachment {
	pub image: Res<ImageView>,
	pub load: Load,
	pub store: bool,
}

pub struct RenderPass<'a, 'frame, 'graph> {
	pub pass: &'a mut PassContext<'frame, 'graph>,
}

impl Drop for RenderPass<'_, '_, '_> {
	fn drop(&mut self) {
		unsafe {
			self.pass.device.device().cmd_end_rendering(self.pass.buf);
		}
	}
}

impl RenderPass<'_, '_, '_> {
	pub fn bind_graphics(&mut self, pipe: &GraphicsPipeline) { pipe.bind(self.pass.device, self.pass.buf); }

	pub fn push(&mut self, offset: usize, value: &impl NoUninit) { self.pass.push(offset, value); }

	pub fn bind_index(&mut self, buf: vk::Buffer, offset: usize, ty: vk::IndexType) {
		unsafe {
			self.pass
				.device
				.device()
				.cmd_bind_index_buffer(self.pass.buf, buf, offset as _, ty);
		}
	}

	pub fn bind_index_res(&mut self, buf: Res<BufferHandle>, offset: usize, ty: vk::IndexType) {
		let buf = self.pass.get(buf).buffer;
		self.bind_index(buf, offset, ty);
	}

	pub fn scissor(&mut self, area: vk::Rect2D) {
		unsafe {
			self.pass.device.device().cmd_set_scissor(self.pass.buf, 0, &[area]);
		}
	}

	pub fn draw_indexed(&mut self, indices: u32, instances: u32, first_index: u32, first_instance: u32) {
		unsafe {
			self.pass.device.device().cmd_draw_indexed(
				self.pass.buf,
				indices,
				instances,
				first_index,
				0,
				first_instance,
			);
		}
	}

	pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
		unsafe {
			self.pass
				.device
				.device()
				.cmd_draw(self.pass.buf, vertices, instances, first_vertex, first_instance);
		}
	}
}
