use ash::vk;

use crate::{
	graph::{
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		ExternalBuffer,
		ExternalImage,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
	},
	resource::{Buffer, BufferHandle, Image, ImageView, Resource, Subresource},
};

#[derive(Copy, Clone)]
pub struct ImageStage {
	pub row_stride: u32,
	pub plane_stride: u32,
	pub subresource: Subresource,
	pub offset: vk::Offset3D,
	pub extent: vk::Extent3D,
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	/// Stage some data into a GPU resource
	///
	/// TODO: Allow using the transfer queue and use a staging buffer instead of an upload buffer.
	pub fn stage_buffer(&mut self, name: &str, dst: Res<BufferHandle>, offset: u64, data: impl AsRef<[u8]> + 'pass) {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		pass.input(
			dst,
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |ctx| Self::exec_buffer_stage(ctx, staging, dst, offset, data.as_ref()));
	}

	pub fn stage_buffer_ext(
		&mut self, name: &str, dst: &Buffer, offset: u64, data: impl AsRef<[u8]> + 'pass,
	) -> Res<BufferHandle> {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		let dst = pass.output(
			ExternalBuffer { handle: dst.handle() },
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |ctx| Self::exec_buffer_stage(ctx, staging, dst, offset, data.as_ref()));
		dst
	}

	/// Stage an image.
	///
	/// Subresource mip count must be 1, and the strides can be 0 to imply tightly packed.
	pub fn stage_image(&mut self, name: &str, dst: Res<ImageView>, stage: ImageStage, data: impl AsRef<[u8]> + 'pass) {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		pass.input(
			dst,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::TransferWrite],
				view_type: None,
				subresource: stage.subresource,
			},
		);
		pass.build(move |ctx| Self::exec_image_stage(ctx, staging, dst, stage, data.as_ref()));
	}

	pub fn stage_image_ext(
		&mut self, name: &str, dst: &Image, stage: ImageStage, data: impl AsRef<[u8]> + 'pass,
	) -> Res<ImageView> {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		let dst = pass.output(
			ExternalImage {
				handle: dst.handle(),
				desc: dst.desc(),
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::TransferWrite],
				view_type: None,
				subresource: stage.subresource,
			},
		);
		pass.build(move |ctx| Self::exec_image_stage(ctx, staging, dst, stage, data.as_ref()));
		dst
	}

	fn make_staging_buffer(pass: &mut PassBuilder, data: &[u8]) -> Res<BufferHandle> {
		pass.output(
			BufferDesc {
				size: data.len() as _,
				upload: true,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferRead],
			},
		)
	}

	fn exec_buffer_stage(
		mut ctx: PassContext, staging: Res<BufferHandle>, dst: Res<BufferHandle>, offset: u64, data: &[u8],
	) {
		let mut staging = ctx.get(staging);
		let dst = ctx.get(dst);
		unsafe {
			staging.data.as_mut().copy_from_slice(data);
			ctx.device.device().cmd_copy_buffer2(
				ctx.buf,
				&vk::CopyBufferInfo2::builder()
					.src_buffer(staging.buffer)
					.dst_buffer(dst.buffer)
					.regions(&[vk::BufferCopy2::builder()
						.src_offset(0)
						.dst_offset(offset)
						.size(data.len() as _)
						.build()]),
			);
		}
	}

	fn exec_image_stage(
		mut ctx: PassContext, staging: Res<BufferHandle>, dst: Res<ImageView>, stage: ImageStage, data: &[u8],
	) {
		let mut staging = ctx.get(staging);
		let dst = ctx.get(dst);
		unsafe {
			assert!(
				stage.subresource.mip_count == 1 || stage.subresource.mip_count == vk::REMAINING_MIP_LEVELS,
				"Only one mip can be staged in a single command"
			);
			staging.data.as_mut().copy_from_slice(data);
			ctx.device.device().cmd_copy_buffer_to_image2(
				ctx.buf,
				&vk::CopyBufferToImageInfo2::builder()
					.src_buffer(staging.buffer)
					.dst_image(dst.image)
					.dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.regions(&[vk::BufferImageCopy2::builder()
						.buffer_offset(0)
						.buffer_row_length(stage.row_stride)
						.buffer_image_height(stage.plane_stride)
						.image_subresource(vk::ImageSubresourceLayers {
							aspect_mask: stage.subresource.aspect,
							mip_level: stage.subresource.first_mip,
							base_array_layer: stage.subresource.first_layer,
							layer_count: stage.subresource.layer_count,
						})
						.image_offset(stage.offset)
						.image_extent(stage.extent)
						.build()]),
			);
		}
	}
}
