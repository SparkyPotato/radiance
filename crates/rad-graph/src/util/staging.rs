use std::alloc::Allocator;

use ash::vk;
use bytemuck::{cast_slice, NoUninit};

use crate::{
	graph::{
		BufferDesc,
		BufferLoc,
		BufferUsage,
		BufferUsageType,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
		VirtualResourceDesc,
	},
	resource::{BufferHandle, ImageView},
	util::pass::ImageCopy,
};

pub struct ByteReader<T, A: Allocator>(pub Vec<T, A>);

impl<T: NoUninit, A: Allocator> AsRef<[u8]> for ByteReader<T, A> {
	fn as_ref(&self) -> &[u8] { cast_slice(&self.0) }
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	/// Stage some data into a GPU resource
	///
	/// TODO: Allow using the transfer queue and use a staging buffer instead of an upload buffer.
	pub fn stage_buffer(&mut self, name: &str, dst: Res<BufferHandle>, offset: u64, data: impl AsRef<[u8]> + 'pass) {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		pass.reference(
			dst,
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |pass| Self::exec_buffer_stage(pass, staging, dst, offset, data.as_ref()));
	}

	pub fn stage_buffer_new<D: VirtualResourceDesc<Resource = BufferHandle>>(
		&mut self, name: &str, dst: D, offset: u64, data: impl AsRef<[u8]> + 'pass,
	) -> Res<BufferHandle> {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		let dst = pass.resource(
			dst,
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |pass| Self::exec_buffer_stage(pass, staging, dst, offset, data.as_ref()));
		dst
	}

	/// Stage an image.
	///
	/// Subresource mip count must be 1, and the strides can be 0 to imply tightly packed.
	pub fn stage_image(&mut self, name: &str, dst: Res<ImageView>, stage: ImageCopy, data: impl AsRef<[u8]> + 'pass) {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		pass.reference(
			dst,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::TransferWrite],
				view_type: None,
				subresource: stage.subresource,
			},
		);
		pass.build(move |pass| Self::exec_image_stage(pass, staging, dst, stage, data.as_ref()));
	}

	pub fn stage_image_new<D: VirtualResourceDesc<Resource = ImageView>>(
		&mut self, name: &str, desc: D, stage: ImageCopy, data: impl AsRef<[u8]> + 'pass,
	) -> Res<ImageView> {
		let mut pass = self.pass(name);
		let staging = Self::make_staging_buffer(&mut pass, data.as_ref());
		let dst = pass.resource(
			desc,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::TransferWrite],
				view_type: None,
				subresource: stage.subresource,
			},
		);
		pass.build(move |pass| Self::exec_image_stage(pass, staging, dst, stage, data.as_ref()));
		dst
	}

	fn make_staging_buffer(pass: &mut PassBuilder, data: &[u8]) -> Res<BufferHandle> {
		pass.resource(
			BufferDesc {
				size: data.len() as _,
				loc: BufferLoc::Upload,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferRead],
			},
		)
	}

	fn exec_buffer_stage(
		mut pass: PassContext, staging: Res<BufferHandle>, dst: Res<BufferHandle>, offset: u64, data: &[u8],
	) {
		pass.write(staging, 0, data);
		pass.copy_buffer(staging, dst, 0, offset as _, data.len());
	}

	fn exec_image_stage(
		mut pass: PassContext, staging: Res<BufferHandle>, dst: Res<ImageView>, stage: ImageCopy, data: &[u8],
	) {
		pass.write(staging, 0, data);
		pass.copy_buffer_to_image(staging, dst, 0, stage);
	}
}
