use ash::vk;
use bytemuck::{bytes_of, cast_slice, from_bytes, NoUninit, Pod};

use crate::{
	arena::IteratorAlloc,
	device::{ComputePipeline, GraphicsPipeline},
	graph::{BufferLoc, PassContext, Res, VirtualResource},
	resource::{BufferHandle, ImageView, Subresource},
};

#[derive(Copy, Clone)]
pub struct ImageCopy {
	pub row_stride: u32,
	pub plane_stride: u32,
	pub subresource: Subresource,
	pub offset: vk::Offset3D,
	pub extent: vk::Extent3D,
}

pub trait ZeroableResource: VirtualResource {
	fn zero(&mut self, pass: &mut PassContext);
}

impl ZeroableResource for BufferHandle {
	fn zero(&mut self, pass: &mut PassContext) {
		unsafe {
			pass.device
				.device()
				.cmd_fill_buffer(pass.buf, self.buffer, 0, vk::WHOLE_SIZE, 0);
		}
	}
}

impl ZeroableResource for ImageView {
	fn zero(&mut self, pass: &mut PassContext) {
		unsafe {
			pass.device.device().cmd_clear_color_image(
				pass.buf,
				self.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&vk::ClearColorValue::default(),
				&[vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(vk::REMAINING_MIP_LEVELS)
					.base_array_layer(0)
					.layer_count(vk::REMAINING_ARRAY_LAYERS)],
			);
		}
	}
}

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

	pub fn zero(&mut self, res: Res<impl ZeroableResource>) {
		let mut res = self.get(res);
		res.zero(self);
	}

	pub fn zero_if_uninit(&mut self, res: Res<impl ZeroableResource>) {
		if self.is_uninit(res) {
			self.zero(res);
		}
	}

	pub fn clear_image(&mut self, res: Res<ImageView>, value: vk::ClearColorValue) {
		unsafe {
			let res = self.get(res);
			self.device.device().cmd_clear_color_image(
				self.buf,
				res.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&value,
				&[vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(vk::REMAINING_MIP_LEVELS)
					.base_array_layer(0)
					.layer_count(vk::REMAINING_ARRAY_LAYERS)],
			);
		}
	}

	pub fn update_buffer(&mut self, res: Res<BufferHandle>, offset: usize, data: &[impl NoUninit]) {
		unsafe {
			let res = self.get(res);
			self.device
				.device()
				.cmd_update_buffer(self.buf, res.buffer, offset as _, cast_slice(data));
		}
	}

	pub fn fill_buffer(&mut self, res: Res<BufferHandle>, data: u32, offset: usize, size: usize) {
		unsafe {
			let res = self.get(res);
			self.device
				.device()
				.cmd_fill_buffer(self.buf, res.buffer, offset as _, size as _, data);
		}
	}

	pub fn copy_buffer(
		&mut self, src: Res<BufferHandle>, dst: Res<BufferHandle>, src_offset: usize, dst_offset: usize, size: usize,
	) {
		unsafe {
			let src = self.get(src);
			let dst = self.get(dst);
			self.device.device().cmd_copy_buffer(
				self.buf,
				src.buffer,
				dst.buffer,
				&[vk::BufferCopy {
					src_offset: src_offset as _,
					dst_offset: dst_offset as _,
					size: size as _,
				}],
			);
		}
	}

	pub fn copy_full_buffer(&mut self, src: Res<BufferHandle>, dst: Res<BufferHandle>, dst_offset: usize) {
		let size = self.desc(src).size;
		self.copy_buffer(src, dst, 0, dst_offset, size as _);
	}

	pub fn copy_buffer_to_image(
		&mut self, src: Res<BufferHandle>, dst: Res<ImageView>, src_offset: usize, copy: ImageCopy,
	) {
		let src = self.get(src);
		let dst = self.get(dst);
		unsafe {
			assert!(
				copy.subresource.mip_count == 1 || copy.subresource.mip_count == vk::REMAINING_MIP_LEVELS,
				"Only one mip can be copied in a single command"
			);
			self.device.device().cmd_copy_buffer_to_image2(
				self.buf,
				&vk::CopyBufferToImageInfo2::default()
					.src_buffer(src.buffer)
					.dst_image(dst.image)
					.dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.regions(&[vk::BufferImageCopy2::default()
						.buffer_offset(src_offset as _)
						.buffer_row_length(copy.row_stride)
						.buffer_image_height(copy.plane_stride)
						.image_subresource(vk::ImageSubresourceLayers {
							aspect_mask: copy.subresource.aspect,
							mip_level: copy.subresource.first_mip,
							base_array_layer: copy.subresource.first_layer,
							layer_count: copy.subresource.layer_count,
						})
						.image_offset(copy.offset)
						.image_extent(copy.extent)]),
			);
		}
	}

	pub fn write(&mut self, res: Res<BufferHandle>, offset: usize, data: &[impl NoUninit]) {
		debug_assert!(
			self.desc(res).loc == BufferLoc::Upload,
			"can only `write` to upload buffers. use `update_buffer` otherwise"
		);
		let res = self.get(res);
		let data: &[u8] = cast_slice(data);
		unsafe {
			std::ptr::copy_nonoverlapping(data.as_ptr(), res.data.as_mut_ptr().add(offset), data.len());
		}
	}

	pub fn write_iter(&mut self, res: Res<BufferHandle>, offset: usize, data: impl IntoIterator<Item = impl NoUninit>) {
		debug_assert!(
			self.desc(res).loc == BufferLoc::Upload,
			"can only `write` to upload buffers. use `update_buffer` otherwise"
		);
		let res = self.get(res);
		unsafe {
			let mut ptr = res.data.as_mut_ptr().add(offset);
			for x in data {
				let data = bytes_of(&x);
				std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
				ptr = ptr.add(data.len());
			}
		}
	}

	pub fn readback<T: Pod>(&mut self, res: Res<BufferHandle>, offset: usize) -> T {
		debug_assert!(
			self.desc(res).loc == BufferLoc::Readback,
			"can only `readback` from readback buffers"
		);
		if self.is_uninit(res) {
			return T::zeroed();
		}
		let res = self.get(res);
		unsafe { *from_bytes(&res.data.as_ref()[offset..][..std::mem::size_of::<T>()]) }
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

	pub fn draw_indexed(
		&mut self, indices: u32, instances: u32, first_index: u32, first_vertex: u32, first_instance: u32,
	) {
		unsafe {
			self.pass.device.device().cmd_draw_indexed(
				self.pass.buf,
				indices,
				instances,
				first_index,
				first_vertex as _,
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
