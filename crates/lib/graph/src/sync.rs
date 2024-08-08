//! A port of from `vk-sync-fork` for radiance.

use std::ops::{BitOr, BitOrAssign};

use ash::vk;
use vk::ImageLayout;

/// Defines all potential shader stages.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Shader {
	Vertex,
	TessellationControl,
	TessellationEvaluation,
	Task,
	Mesh,
	Geometry,
	Fragment,
	Compute,
	RayTracing,
	/// Any or all shader stages.
	Any,
}

#[rustfmt::skip]
macro_rules! gen_usage_enums {
	(
		pub enum BufferOnlyUsage { $($(#[$buffer_doc:meta])* $buffer_only_name:ident $(( $buffer_only_tt:tt ))?,)* };
		pub enum ImageOnlyUsage { $($(#[$image_doc:meta])* $image_only_name:ident $(( $image_only_tt:tt ))?,)* };
		pub enum CommonUsage { $($(#[$common_doc:meta])* $common_name:ident $(( $common_tt:tt ))?,)* };
	) => {
		/// Defines all potential buffer usages.
		#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
		pub enum BufferUsage {
			$($(#[$buffer_doc])* $buffer_only_name $(( $buffer_only_tt ))?,)*
			$($(#[$common_doc])* $common_name $(( $common_tt ))?,)*
		}

		/// Defines all potential image usages.
		#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
		pub enum ImageUsage {
			$($(#[$image_doc])* $image_only_name $(( $image_only_tt ))?,)*
			$($(#[$common_doc])* $common_name $(( $common_tt ))?,)*
		}

		/// Defines all potential resource usages.
		#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
		pub enum UsageType {
			$($(#[$buffer_doc])* $buffer_only_name $(( $buffer_only_tt ))?,)*
			$($(#[$image_doc])* $image_only_name $(( $image_only_tt ))?,)*
			$($(#[$common_doc])* $common_name $(( $common_tt ))?,)*
		}

		#[allow(non_snake_case)]
		impl From<BufferUsage> for UsageType {
			fn from(usage: BufferUsage) -> Self {
				match usage {
					$(BufferUsage::$buffer_only_name $(( $buffer_only_tt ))? => Self::$buffer_only_name $(( $buffer_only_tt ))?,)*
					$(BufferUsage::$common_name $(( $common_tt ))? => Self::$common_name $(( $common_tt ))?,)*
				}
			}
		}

		#[allow(non_snake_case)]
		impl From<ImageUsage> for UsageType {
			fn from(usage: ImageUsage) -> Self {
				match usage {
					$(ImageUsage::$image_only_name $(( $image_only_tt ))? => Self::$image_only_name $(( $image_only_tt ))?,)*
					$(ImageUsage::$common_name $(( $common_tt ))? => Self::$common_name $(( $common_tt ))?,)*
				}
			}
		}
	};
}

gen_usage_enums! {
	pub enum BufferOnlyUsage {
		/// Read as an indirect buffer for drawing or dispatch.
		IndirectBuffer,
		/// Read as an index buffer for drawing.
		IndexBuffer,
		/// Read as a vertex buffer for drawing.
		VertexBuffer,
		/// Read as a uniform buffer in a shader.
		ShaderReadUniformBuffer(Shader),
		/// Read on the host.
		HostRead,
		/// Written on the host. Do not use for host writes before a submit - they are already synchronized.
		HostWrite,
		/// Read when building acceleration structures.
		AccelerationStructureBuildRead,
		/// Written when building acceleration structures.
		AccelerationStructureBuildWrite,
		/// Written as scratch data during acceleration structure build.
		AccelerationStructureBuildScratch,
	};

	pub enum ImageOnlyUsage {
		/// Read as a sampled image in a shader.
		ShaderReadSampledImage(Shader),
		/// Read by blending/logic operations or subpass load operations.
		ColorAttachmentRead,
		/// Read by depth/stencil tests or subpass load operations.
		DepthStencilAttachmentRead,
		/// Read by the presentation engine (i.e. `vkQueuePresentKHR`).
		Present,
		/// Written as a color attachment during rendering, or via a subpass store op.
		ColorAttachmentWrite,
		/// Written as a depth/stencil attachment during rendering, or via a subpass store op.
		DepthStencilAttachmentWrite,
		/// Override the default layout decided.
		CustomLayout(ImageLayout),
	};

	pub enum CommonUsage {
		/// No access. Useful primarily for initialization.
		Nothing,
		/// Read as a storage resource in a shader.
		ShaderStorageRead(Shader),
		/// Read as the source of a transfer operation.
		TransferRead,
		/// Written as a storage resource in a shader.
		ShaderStorageWrite(Shader),
		/// Written as the destination of a transfer operation.
		TransferWrite,
		/// Covers any access - useful for debug, generally avoid for performance reasons.
		#[default]
		General,
	};
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct AccessInfo {
	pub stage_mask: vk::PipelineStageFlags2,
	pub access_mask: vk::AccessFlags2,
	pub image_layout: vk::ImageLayout,
}

impl BitOr for AccessInfo {
	type Output = Self;

	fn bitor(self, rhs: Self) -> Self::Output {
		Self {
			stage_mask: self.stage_mask | rhs.stage_mask,
			access_mask: self.access_mask | rhs.access_mask,
			image_layout: {
				debug_assert!(
					self.image_layout == rhs.image_layout
						|| self.image_layout == vk::ImageLayout::UNDEFINED
						|| rhs.image_layout == vk::ImageLayout::UNDEFINED,
					"Cannot merge `AccessInfo`s with different layouts ({:?} and {:?})",
					self.image_layout,
					rhs.image_layout,
				);
				rhs.image_layout
			},
		}
	}
}

impl BitOrAssign for AccessInfo {
	fn bitor_assign(&mut self, rhs: Self) { *self = *self | rhs; }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
	fn from(usage: BufferUsage) -> Self {
		match usage {
			BufferUsage::IndirectBuffer => vk::BufferUsageFlags::INDIRECT_BUFFER,
			BufferUsage::IndexBuffer => vk::BufferUsageFlags::INDEX_BUFFER,
			BufferUsage::VertexBuffer => vk::BufferUsageFlags::VERTEX_BUFFER,
			BufferUsage::ShaderReadUniformBuffer(_) => vk::BufferUsageFlags::UNIFORM_BUFFER,
			BufferUsage::Nothing => vk::BufferUsageFlags::empty(),
			BufferUsage::ShaderStorageRead(_) => vk::BufferUsageFlags::STORAGE_BUFFER,
			BufferUsage::TransferRead => vk::BufferUsageFlags::TRANSFER_SRC,
			BufferUsage::HostRead => vk::BufferUsageFlags::empty(),
			BufferUsage::ShaderStorageWrite(_) => vk::BufferUsageFlags::STORAGE_BUFFER,
			BufferUsage::TransferWrite => vk::BufferUsageFlags::TRANSFER_DST,
			BufferUsage::HostWrite => vk::BufferUsageFlags::empty(),
			BufferUsage::General => vk::BufferUsageFlags::empty(),
			BufferUsage::AccelerationStructureBuildRead => {
				vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
			},
			BufferUsage::AccelerationStructureBuildWrite => vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
			BufferUsage::AccelerationStructureBuildScratch => vk::BufferUsageFlags::STORAGE_BUFFER,
		}
	}
}

impl From<ImageUsage> for vk::ImageUsageFlags {
	fn from(usage: ImageUsage) -> Self {
		match usage {
			ImageUsage::ShaderReadSampledImage(_) => vk::ImageUsageFlags::SAMPLED,
			ImageUsage::ColorAttachmentRead => vk::ImageUsageFlags::COLOR_ATTACHMENT,
			ImageUsage::DepthStencilAttachmentRead => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
			ImageUsage::Present => vk::ImageUsageFlags::empty(),
			ImageUsage::ColorAttachmentWrite => vk::ImageUsageFlags::COLOR_ATTACHMENT,
			ImageUsage::DepthStencilAttachmentWrite => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
			ImageUsage::Nothing => vk::ImageUsageFlags::empty(),
			ImageUsage::ShaderStorageRead(_) => vk::ImageUsageFlags::STORAGE,
			ImageUsage::TransferRead => vk::ImageUsageFlags::TRANSFER_SRC,
			ImageUsage::ShaderStorageWrite(_) => vk::ImageUsageFlags::STORAGE,
			ImageUsage::TransferWrite => vk::ImageUsageFlags::TRANSFER_DST,
			ImageUsage::General => vk::ImageUsageFlags::empty(),
			ImageUsage::CustomLayout(_) => vk::ImageUsageFlags::empty(),
		}
	}
}

impl From<UsageType> for AccessInfo {
	fn from(usage: UsageType) -> Self {
		match usage {
			UsageType::Nothing => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::NONE,
				access_mask: vk::AccessFlags2::NONE,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::IndirectBuffer => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT,
				access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::IndexBuffer => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::INDEX_INPUT,
				access_mask: vk::AccessFlags2::INDEX_READ,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::VertexBuffer => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
				access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::ShaderReadUniformBuffer(s) => AccessInfo {
				stage_mask: get_pipeline_stage(s),
				access_mask: vk::AccessFlags2::UNIFORM_READ,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::ShaderReadSampledImage(s) => AccessInfo {
				stage_mask: get_pipeline_stage(s),
				access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
				image_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
			},
			UsageType::ColorAttachmentRead => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
				access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
				image_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
			},
			UsageType::DepthStencilAttachmentRead => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
					| vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
				access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
				image_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
			},
			UsageType::ShaderStorageRead(s) => AccessInfo {
				stage_mask: get_pipeline_stage(s),
				access_mask: vk::AccessFlags2::SHADER_STORAGE_READ,
				image_layout: vk::ImageLayout::GENERAL,
			},
			UsageType::TransferRead => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::TRANSFER,
				access_mask: vk::AccessFlags2::TRANSFER_READ,
				image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			},
			UsageType::HostRead => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::HOST,
				access_mask: vk::AccessFlags2::HOST_READ,
				image_layout: vk::ImageLayout::GENERAL,
			},
			UsageType::Present => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::empty(),
				access_mask: vk::AccessFlags2::empty(),
				image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
			},
			UsageType::ShaderStorageWrite(s) => AccessInfo {
				stage_mask: get_pipeline_stage(s),
				access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
				image_layout: vk::ImageLayout::GENERAL,
			},
			UsageType::ColorAttachmentWrite => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
				access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
				image_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
			},
			UsageType::DepthStencilAttachmentWrite => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
					| vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
				access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
				image_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
			},
			UsageType::TransferWrite => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::TRANSFER,
				access_mask: vk::AccessFlags2::TRANSFER_WRITE,
				image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			},
			UsageType::HostWrite => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::HOST,
				access_mask: vk::AccessFlags2::HOST_WRITE,
				image_layout: vk::ImageLayout::GENERAL,
			},
			UsageType::General => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
				access_mask: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
				image_layout: vk::ImageLayout::GENERAL,
			},
			UsageType::AccelerationStructureBuildRead => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
				access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::AccelerationStructureBuildWrite => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
				access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::AccelerationStructureBuildScratch => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
				access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
				image_layout: vk::ImageLayout::UNDEFINED,
			},
			UsageType::CustomLayout(image_layout) => AccessInfo {
				stage_mask: vk::PipelineStageFlags2::empty(),
				access_mask: vk::AccessFlags2::empty(),
				image_layout,
			},
		}
	}
}

impl From<Shader> for vk::PipelineStageFlags2 {
	fn from(shader: Shader) -> Self {
		match shader {
			Shader::Vertex => vk::PipelineStageFlags2::VERTEX_SHADER,
			Shader::TessellationControl => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
			Shader::TessellationEvaluation => vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
			Shader::Task => vk::PipelineStageFlags2::TASK_SHADER_EXT,
			Shader::Mesh => vk::PipelineStageFlags2::MESH_SHADER_EXT,
			Shader::Geometry => vk::PipelineStageFlags2::GEOMETRY_SHADER,
			Shader::Fragment => vk::PipelineStageFlags2::FRAGMENT_SHADER,
			Shader::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
			Shader::Any => vk::PipelineStageFlags2::ALL_GRAPHICS,
			Shader::RayTracing => vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
		}
	}
}

impl BufferUsage {
	pub fn into_usage(self) -> UsageType { self.into() }
}

impl ImageUsage {
	pub fn into_usage(self) -> UsageType { self.into() }
}

pub fn get_access_info(usage: UsageType) -> AccessInfo { usage.into() }

pub fn is_write_access(usage: UsageType) -> bool {
	matches!(
		usage,
		UsageType::ShaderStorageWrite(_)
			| UsageType::ColorAttachmentWrite
			| UsageType::DepthStencilAttachmentWrite
			| UsageType::TransferWrite
			| UsageType::HostWrite
			| UsageType::General
			| UsageType::AccelerationStructureBuildWrite
			| UsageType::AccelerationStructureBuildScratch
	)
}

fn get_pipeline_stage(shader: Shader) -> vk::PipelineStageFlags2 { shader.into() }

/// Image barriers should only be used when a queue family ownership transfer
/// or an image layout transition is required - prefer global barriers at all
/// other times.
///
/// Usage types are defined in the same way as for a global memory barrier, but
/// they only affect the image subresource range identified by `image` and
/// `range`, rather than all resources.
///
/// `src_queue_family_index`, `dst_queue_family_index`, `image`, and `range` will
/// be passed unmodified into an image memory barrier.
///
/// An image barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them. The release barrier must also have no `next_access`, and the acquire barrier must have
/// no `previous_access`.
///
/// If `discard_contents` is set to true, the contents of the image become
/// undefined after the barrier is executed, which can result in a performance
/// boost over attempting to preserve the contents.
#[derive(Debug, Copy, Clone)]
pub struct ImageBarrier<'a> {
	pub previous_usages: &'a [UsageType],
	pub next_usages: &'a [UsageType],
	pub discard_contents: bool,
	pub image: vk::Image,
	pub range: vk::ImageSubresourceRange,
}

impl Default for ImageBarrier<'_> {
	fn default() -> Self {
		Self {
			previous_usages: &[],
			next_usages: &[],
			discard_contents: false,
			image: vk::Image::null(),
			range: vk::ImageSubresourceRange::default(),
		}
	}
}

impl From<ImageBarrier<'_>> for vk::ImageMemoryBarrier2<'static> {
	fn from(barrier: ImageBarrier) -> Self {
		let previous_access = as_previous_access(barrier.previous_usages.iter().copied(), barrier.discard_contents);
		let mut next_access = as_next_access(barrier.next_usages.iter().copied(), previous_access);

		// Ensure that the stage masks are valid if no stages were determined
		if next_access.stage_mask == vk::PipelineStageFlags2::empty() {
			next_access.stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
		}

		ImageBarrierAccess {
			image: barrier.image,
			range: barrier.range,
			previous_access,
			next_access,
		}
		.into()
	}
}

/// Image barriers should only be used when a queue family ownership transfer
/// or an image layout transition is required - prefer global barriers at all
/// other times.
///
/// Usage types are defined in the same way as for a global memory barrier, but
/// they only affect the image subresource range identified by `image` and
/// `range`, rather than all resources.
///
/// `src_queue_family_index`, `dst_queue_family_index`, `image`, and `range` will
/// be passed unmodified into an image memory barrier.
///
/// An image barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them. The release barrier must also have no `next_access`, and the acquire barrier must have
/// no `previous_access`.
///
/// If `discard_contents` is set to true, the contents of the image become
/// undefined after the barrier is executed, which can result in a performance
/// boost over attempting to preserve the contents.
#[derive(Debug, Copy, Clone)]
pub struct ImageBarrierAccess {
	pub previous_access: AccessInfo,
	pub next_access: AccessInfo,
	pub image: vk::Image,
	pub range: vk::ImageSubresourceRange,
}

impl Default for ImageBarrierAccess {
	fn default() -> Self {
		Self {
			previous_access: Default::default(),
			next_access: Default::default(),
			image: vk::Image::null(),
			range: vk::ImageSubresourceRange::default(),
		}
	}
}

impl From<ImageBarrierAccess> for vk::ImageMemoryBarrier2<'static> {
	fn from(barrier: ImageBarrierAccess) -> Self {
		vk::ImageMemoryBarrier2 {
			image: barrier.image,
			subresource_range: barrier.range,
			src_stage_mask: barrier.previous_access.stage_mask,
			src_access_mask: barrier.previous_access.access_mask,
			old_layout: barrier.previous_access.image_layout,
			dst_stage_mask: barrier.next_access.stage_mask,
			dst_access_mask: barrier.next_access.access_mask,
			new_layout: barrier.next_access.image_layout,
			src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
			dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
			..Default::default()
		}
	}
}

/// Global barriers define a set of accesses on multiple resources at once.
/// If a buffer or image doesn't require a queue ownership transfer, or an image
/// doesn't require a layout transition, a global barrier should be preferred.
///
/// Simply define the previous and next usage types of resources affected.
#[derive(Debug, Default, Copy, Clone)]
pub struct GlobalBarrier<'a> {
	pub previous_usages: &'a [UsageType],
	pub next_usages: &'a [UsageType],
}

impl From<GlobalBarrier<'_>> for vk::MemoryBarrier2<'static> {
	fn from(barrier: GlobalBarrier<'_>) -> Self {
		let previous_access = as_previous_access(barrier.previous_usages.iter().copied(), false);
		let mut next_access = as_next_access(barrier.next_usages.iter().copied(), previous_access);

		// Ensure that the stage masks are valid if no stages were determined
		if next_access.stage_mask == vk::PipelineStageFlags2::empty() {
			next_access.stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
		}

		GlobalBarrierAccess {
			previous_access,
			next_access,
		}
		.into()
	}
}

/// Global barriers define a set of accesses on multiple resources at once.
/// If a buffer or image doesn't require a queue ownership transfer, or an image
/// doesn't require a layout transition, a global barrier should be preferred.
///
/// Simply define the previous and next usage types of resources affected.
#[derive(Debug, Default, Copy, Clone)]
pub struct GlobalBarrierAccess {
	pub previous_access: AccessInfo,
	pub next_access: AccessInfo,
}

impl From<GlobalBarrierAccess> for vk::MemoryBarrier2<'static> {
	fn from(barrier: GlobalBarrierAccess) -> Self {
		vk::MemoryBarrier2 {
			src_stage_mask: barrier.previous_access.stage_mask,
			src_access_mask: barrier.previous_access.access_mask,
			dst_stage_mask: barrier.next_access.stage_mask,
			dst_access_mask: barrier.next_access.access_mask,
			..Default::default()
		}
	}
}

pub fn get_image_barrier(image_barrier: &ImageBarrier) -> vk::ImageMemoryBarrier2<'static> { (*image_barrier).into() }
pub fn get_image_barrier_access(image_barrier: &ImageBarrierAccess) -> vk::ImageMemoryBarrier2 {
	(*image_barrier).into()
}

pub fn get_global_barrier(global_barrier: &GlobalBarrier) -> vk::MemoryBarrier2<'static> { (*global_barrier).into() }
pub fn get_global_barrier_access(global_barrier: &GlobalBarrierAccess) -> vk::MemoryBarrier2 {
	(*global_barrier).into()
}

impl UsageType {
	pub fn into_prev_access(self, discard_contents: bool) -> AccessInfo {
		let mut info = get_access_info(self);

		// We don't care about previous reads, only writes need to be made available.
		if !is_write_access(self) {
			info.access_mask = vk::AccessFlags2::NONE;
		}
		// If we're discarding contents, we don't care about the previous layout.
		if discard_contents {
			info.image_layout = vk::ImageLayout::UNDEFINED;
		}

		info
	}

	pub fn into_next_access(self, prev_access: AccessInfo) -> AccessInfo {
		let mut info = get_access_info(self);

		// If the previous access was a read, and no layout transition in required, we don't need to make anything
		// available.
		if prev_access.access_mask == vk::AccessFlags2::NONE && prev_access.image_layout == info.image_layout {
			info.access_mask = vk::AccessFlags2::NONE;
		}

		info
	}
}

pub fn as_previous_access(usages: impl IntoIterator<Item = UsageType>, discard_contents: bool) -> AccessInfo {
	usages
		.into_iter()
		.map(|u| u.into_prev_access(discard_contents))
		.reduce(|a, b| a | b)
		.unwrap_or_default()
}

pub fn as_next_access(usages: impl IntoIterator<Item = UsageType>, prev_access: AccessInfo) -> AccessInfo {
	usages
		.into_iter()
		.map(|u| u.into_next_access(prev_access))
		.reduce(|a, b| a | b)
		.unwrap_or_default()
}
