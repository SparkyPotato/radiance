use std::{hash::BuildHasherDefault, hint::unreachable_unchecked, ptr::NonNull};

use ash::vk;

pub use crate::sync::{BufferUsage as BufferUsageType, ImageUsage as ImageUsageType, Shader};
use crate::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{compile::Resource, ArenaMap, Caches, ReadId},
	resource::{GpuBufferHandle, ImageView, ImageViewDesc, ImageViewUsage, UploadBufferHandle},
	sync::UsageType,
};

/// A description for a buffer for uploading data from the CPU to the GPU.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct UploadBufferDesc {
	pub size: u64,
}

/// A description for a GPU buffer.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct GpuBufferDesc {
	pub size: u64,
}

/// The usage of a buffer in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct BufferUsage<'a> {
	pub usages: &'a [BufferUsageType],
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferUsageOwned<'graph> {
	pub usages: Vec<BufferUsageType, &'graph Arena>,
}

/// A description for an image.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageDesc {
	pub size: vk::Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
}

/// The usage of an image in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsage<'a> {
	/// The format to view the image as. This can be different from the format in [`ImageDesc`], but must be
	/// [compatible].
	///
	/// [compatible]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#formats-compatibility-classes
	pub format: vk::Format,
	pub usages: &'a [ImageUsageType],
	pub view_type: vk::ImageViewType,
	pub aspect: vk::ImageAspectFlags,
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsageOwned<'graph> {
	pub format: vk::Format,
	pub usages: Vec<ImageUsageType, &'graph Arena>,
	pub view_type: vk::ImageViewType,
	pub aspect: vk::ImageAspectFlags,
}

impl ImageUsageOwned<'_> {
	pub fn create_flags(&self) -> vk::ImageCreateFlags {
		match self.view_type {
			vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => vk::ImageCreateFlags::CUBE_COMPATIBLE,
			vk::ImageViewType::TYPE_2D_ARRAY => vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
			_ => vk::ImageCreateFlags::empty(),
		}
	}
}

/// Synchronization regarding an external resource.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ExternalSync<U> {
	/// The semaphore to wait on or signal. If no cross-queue sync is required, this is `::null()`.
	pub semaphore: vk::Semaphore,
	/// If `semaphore` is a timeline semaphore, the value to wait on or set.
	pub value: u64,
	/// The related usage of the resource.
	pub usage: U,
	/// Other queue. Ownership will be transferred to the graphics queue if it is a `wait`, or transferred to this
	/// queue if it is a `signal`.
	pub queue: Option<u32>,
}

impl<U> ExternalSync<U> {
	pub(crate) fn map<F, T>(&self, f: F) -> ExternalSync<T>
	where
		F: FnOnce(&U) -> T,
	{
		ExternalSync {
			semaphore: self.semaphore,
			value: self.value,
			usage: f(&self.usage),
			queue: self.queue,
		}
	}
}

/// A buffer external to the render graph.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBuffer<'a> {
	/// The handle to the buffer. This is passed as-is to the render pass.
	pub handle: GpuBufferHandle,
	/// The external usage of the buffer before the render pass is executed.
	pub prev_usage: Option<ExternalSync<&'a [BufferUsageType]>>,
	/// The external usage of the buffer after the render pass is executed. This is usually not required.
	pub next_usage: Option<ExternalSync<&'a [BufferUsageType]>>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBufferOwned<'graph> {
	pub handle: GpuBufferHandle,
	pub prev_usage: Option<ExternalSync<Vec<BufferUsageType, &'graph Arena>>>,
	pub next_usage: Option<ExternalSync<Vec<BufferUsageType, &'graph Arena>>>,
}

/// An image external to the render graph.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalImage<'a> {
	/// The image. Image views are created to this image.
	pub handle: ash::vk::Image,
	/// The external usage of the image before the render pass is executed.
	pub prev_usage: Option<ExternalSync<&'a [ImageUsageType]>>,
	/// The external usage of the image after the render pass is executed. This is usually not required.
	pub next_usage: Option<ExternalSync<&'a [ImageUsageType]>>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalImageOwned<'graph> {
	pub handle: ash::vk::Image,
	pub prev_usage: Option<ExternalSync<Vec<ImageUsageType, &'graph Arena>>>,
	pub next_usage: Option<ExternalSync<Vec<ImageUsageType, &'graph Arena>>>,
}

pub trait ToOwnedArena {
	type Owned<'a>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a>;
}

impl ToOwnedArena for &'_ [BufferUsageType] {
	type Owned<'a> = Vec<BufferUsageType, &'a Arena>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> { self.iter().copied().collect_in(arena) }
}

impl ToOwnedArena for &'_ [ImageUsageType] {
	type Owned<'a> = Vec<ImageUsageType, &'a Arena>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> { self.iter().copied().collect_in(arena) }
}

impl ToOwnedArena for BufferUsage<'_> {
	type Owned<'a> = BufferUsageOwned<'a>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> {
		BufferUsageOwned {
			usages: self.usages.to_owned_arena(arena),
		}
	}
}

impl ToOwnedArena for ImageUsage<'_> {
	type Owned<'a> = ImageUsageOwned<'a>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> {
		ImageUsageOwned {
			format: self.format,
			usages: self.usages.to_owned_arena(arena),
			view_type: self.view_type,
			aspect: self.aspect,
		}
	}
}

impl ToOwnedArena for ExternalSync<&'_ [BufferUsageType]> {
	type Owned<'a> = ExternalSync<Vec<BufferUsageType, &'a Arena>>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> { self.map(|usage| usage.to_owned_arena(arena)) }
}

impl ToOwnedArena for ExternalSync<&'_ [ImageUsageType]> {
	type Owned<'a> = ExternalSync<Vec<ImageUsageType, &'a Arena>>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> { self.map(|usage| usage.to_owned_arena(arena)) }
}

impl ToOwnedArena for ExternalBuffer<'_> {
	type Owned<'a> = ExternalBufferOwned<'a>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> {
		ExternalBufferOwned {
			handle: self.handle,
			prev_usage: self.prev_usage.map(|x| x.to_owned_arena(arena)),
			next_usage: self.next_usage.map(|x| x.to_owned_arena(arena)),
		}
	}
}

impl ToOwnedArena for ExternalImage<'_> {
	type Owned<'a> = ExternalImageOwned<'a>;

	fn to_owned_arena<'a>(&self, arena: &'a Arena) -> Self::Owned<'a> {
		ExternalImageOwned {
			handle: self.handle,
			prev_usage: self.prev_usage.map(|x| x.to_owned_arena(arena)),
			next_usage: self.next_usage.map(|x| x.to_owned_arena(arena)),
		}
	}
}

pub trait Usage {
	type Inner: Copy + Into<UsageType>;

	fn get_usages(&self) -> &[Self::Inner];
}

impl Usage for BufferUsageOwned<'_> {
	type Inner = BufferUsageType;

	fn get_usages(&self) -> &[Self::Inner] { &self.usages }
}

impl Usage for ImageUsageOwned<'_> {
	type Inner = ImageUsageType;

	fn get_usages(&self) -> &[Self::Inner] { &self.usages }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ResourceLifetime {
	pub start: u32,
	pub end: u32,
}

impl ResourceLifetime {
	pub fn singular(pass: u32) -> Self { Self { start: pass, end: pass } }

	pub fn union(self, other: Self) -> Self {
		Self {
			start: self.start.min(other.start),
			end: self.end.max(other.end),
		}
	}

	pub fn independent(self, other: Self) -> bool { self.start > other.end || self.end < other.start }
}

#[derive(Clone)]
pub struct VirtualResourceData<'graph> {
	pub lifetime: ResourceLifetime,
	pub ty: VirtualResourceType<'graph>,
}

pub trait VirtualResourceDesc {
	type Resource: VirtualResource;

	fn ty<'graph>(
		self, write_usage: <Self::Resource as VirtualResource>::Usage<'_>, arena: &'graph Arena,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph>;
}

pub trait VirtualResource {
	type Usage<'a>;

	unsafe fn from_res(pass: u32, res: &Resource, caches: &mut Caches, device: &Device) -> Self;

	unsafe fn add_read_usage(ty: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>);
}

#[derive(Clone)]
pub struct GpuData<'graph, T, U> {
	pub desc: T,
	pub write_usage: U,
	pub read_usages: ArenaMap<'graph, u32, U>,
}

#[derive(Clone)]
pub enum GpuBufferType<'graph> {
	Internal(u64),
	External(ExternalBufferOwned<'graph>),
}

#[derive(Clone)]
pub enum ImageType<'graph> {
	Internal(ImageDesc),
	External(ExternalImageOwned<'graph>),
}

#[derive(Clone)]
pub enum VirtualResourceType<'graph> {
	Data(NonNull<()>),
	UploadBuffer(GpuData<'graph, u64, BufferUsageOwned<'graph>>),
	GpuBuffer(GpuData<'graph, GpuBufferType<'graph>, BufferUsageOwned<'graph>>),
	Image(GpuData<'graph, ImageType<'graph>, ImageUsageOwned<'graph>>),
}

impl<'graph> VirtualResourceType<'graph> {
	unsafe fn upload_buffer(&mut self) -> &mut GpuData<'graph, u64, BufferUsageOwned<'graph>> {
		match self {
			VirtualResourceType::UploadBuffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn gpu_buffer(&mut self) -> &mut GpuData<'graph, GpuBufferType<'graph>, BufferUsageOwned<'graph>> {
		match self {
			VirtualResourceType::GpuBuffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&mut self) -> &mut GpuData<'graph, ImageType<'graph>, ImageUsageOwned<'graph>> {
		match self {
			VirtualResourceType::Image(data) => data,
			_ => unreachable_unchecked(),
		}
	}
}

impl VirtualResource for UploadBufferHandle {
	type Usage<'a> = BufferUsage<'a>;

	unsafe fn from_res(_: u32, res: &Resource, _: &mut Caches, _: &Device) -> Self { res.upload_buffer() }

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>) {
		let u = &mut res.ty.upload_buffer().read_usages;
		u.insert(pass, usage.to_owned_arena(u.allocator()));
	}
}

impl VirtualResourceDesc for UploadBufferDesc {
	type Resource = UploadBufferHandle;

	fn ty<'graph>(
		self, write_usage: BufferUsage<'_>, arena: &'graph Arena, _: &mut Vec<VirtualResourceData, &Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::UploadBuffer(GpuData {
			desc: self.size,
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResource for GpuBufferHandle {
	type Usage<'a> = BufferUsage<'a>;

	unsafe fn from_res(_: u32, res: &Resource, _: &mut Caches, _: &Device) -> Self { res.gpu_buffer().resource.handle }

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>) {
		let u = &mut res.ty.gpu_buffer().read_usages;
		u.insert(pass, usage.to_owned_arena(u.allocator()));
	}
}

impl VirtualResourceDesc for GpuBufferDesc {
	type Resource = GpuBufferHandle;

	fn ty<'graph>(
		self, write_usage: BufferUsage<'_>, arena: &'graph Arena,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::GpuBuffer(GpuData {
			desc: GpuBufferType::Internal(self.size),
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResource for ImageView {
	type Usage<'a> = ImageUsage<'a>;

	unsafe fn from_res(pass: u32, res: &Resource, caches: &mut Caches, device: &Device) -> Self {
		let res = &res.image().resource;
		let usage = &res.usages[&pass];

		caches
			.image_views
			.get(
				device,
				ImageViewDesc {
					image: res.handle,
					view_type: usage.view_type,
					format: usage.format,
					usage: {
						let mut u = ImageViewUsage::None;
						for i in usage.usages.iter() {
							u = u | match i {
								ImageUsageType::ShaderStorageRead(_) => ImageViewUsage::Storage,
								ImageUsageType::ShaderStorageWrite(_) => ImageViewUsage::Storage,
								ImageUsageType::ShaderReadSampledImage(_) => ImageViewUsage::Sampled,
								ImageUsageType::General => ImageViewUsage::Both,
								_ => ImageViewUsage::None,
							};
						}
						u
					},
					aspect: usage.aspect,
				},
			)
			.expect("Failed to create image view")
	}

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>) {
		let image = res.ty.image();
		debug_assert!(
			compatible_formats(image.write_usage.format, usage.format),
			"`{:?}` and `{:?}` are not compatible",
			image.write_usage.format,
			usage.format
		);
		image
			.read_usages
			.insert(pass, usage.to_owned_arena(image.read_usages.allocator()));
	}
}

impl VirtualResourceDesc for ImageDesc {
	type Resource = ImageView;

	fn ty<'graph>(
		self, usage: ImageUsage<'_>, arena: &'graph Arena, _: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		_: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(GpuData {
			desc: ImageType::Internal(self),
			write_usage: usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ExternalBuffer<'_> {
	type Resource = GpuBufferHandle;

	fn ty<'graph>(
		self, write_usage: BufferUsage<'_>, arena: &'graph Arena,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::GpuBuffer(GpuData {
			desc: GpuBufferType::External(self.to_owned_arena(arena)),
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ExternalImage<'_> {
	type Resource = ImageView;

	fn ty<'graph>(
		self, write_usage: ImageUsage<'_>, arena: &'graph Arena,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(GpuData {
			desc: ImageType::External(self.to_owned_arena(arena)),
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ReadId<ImageView> {
	type Resource = ImageView;

	fn ty<'graph>(
		self, write_usage: ImageUsage<'_>, arena: &'graph Arena,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(GpuData {
			desc: unsafe { resources[self.id - base_id].ty.image().desc.clone() },
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ReadId<GpuBufferHandle> {
	type Resource = GpuBufferHandle;

	fn ty<'graph>(
		self, write_usage: BufferUsage<'_>, arena: &'graph Arena,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::GpuBuffer(GpuData {
			desc: unsafe { resources[self.id - base_id].ty.gpu_buffer().desc.clone() },
			write_usage: write_usage.to_owned_arena(arena),
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

pub fn compatible_formats(a: vk::Format, b: vk::Format) -> bool { get_format_block(a) == get_format_block(b) }

fn get_format_block(f: vk::Format) -> i32 {
	macro_rules! f {
		($raw:ident,($i:ident)) => {
			vk::Format::$i.as_raw() == $raw
		};
		($raw:ident,($f:ident : $t:ident)) => {
			(vk::Format::$f.as_raw()..=vk::Format::$t.as_raw()).contains(&$raw)
		};
	}

	macro_rules! select {
		($raw:ident, $($rest:tt)*) => {
			select!(# $raw, 0, $($rest)*)
		};

	    (# $raw:ident, $v:expr, $($tt:tt)||+, $($rest:tt)*) => {
			if $(f!($raw, $tt))||* {
				$v
			} else {
				select!(# $raw, $v + 1, $($rest)*)
			}
		};

		(# $raw:ident, $v:expr,) => {
			{
				$raw
			}
		};
	}

	let raw = f.as_raw();

	select! {
		raw,
		(R4G4_UNORM_PACK8) || (R8_UNORM:R8_SRGB),
		(R4G4B4A4_UNORM_PACK16:A1R5G5B5_UNORM_PACK16) || (R8G8_UNORM:R8G8_SRGB) || (R16_UNORM:R16_SFLOAT),
		(R8G8B8_UNORM:B8G8R8_SRGB),
		(R10X6G10X6_UNORM_2PACK16) || (R12X4G12X4_UNORM_2PACK16) || (R8G8B8A8_UNORM:A2B10G10R10_SINT_PACK32)
		|| (R16G16_UNORM:R16G16_SFLOAT) || (R32_UINT:R32_SFLOAT) || (B10G11R11_UFLOAT_PACK32:E5B9G9R9_UFLOAT_PACK32),
		(R16G16B16_UNORM:R16G16B16_SFLOAT),
		(R16G16B16A16_UNORM:R16G16B16A16_SFLOAT) || (R32G32_UINT:R32G32_SFLOAT) || (R64_UINT:R64_SFLOAT),
		(R32G32B32_UINT:R32G32B32_SFLOAT),
		(R32G32B32A32_UINT:R32G32B32A32_SFLOAT) || (R64G64_UINT:R64G64_SFLOAT),
		(R64G64B64_UINT:R64G64B64_SFLOAT),
		(R64G64B64A64_UINT:R64G64B64A64_SFLOAT),
		(BC1_RGB_UNORM_BLOCK:BC1_RGB_SRGB_BLOCK),
		(BC1_RGBA_UNORM_BLOCK:BC1_RGBA_SRGB_BLOCK),
		(BC2_UNORM_BLOCK:BC2_SRGB_BLOCK),
		(BC3_UNORM_BLOCK:BC3_SRGB_BLOCK),
		(BC4_UNORM_BLOCK:BC4_SNORM_BLOCK),
		(BC5_UNORM_BLOCK:BC5_SNORM_BLOCK),
		(BC6H_UFLOAT_BLOCK:BC6H_SFLOAT_BLOCK),
		(BC7_UNORM_BLOCK:BC7_SRGB_BLOCK),
		(ETC2_R8G8B8_UNORM_BLOCK:ETC2_R8G8B8_SRGB_BLOCK),
		(ETC2_R8G8B8A1_UNORM_BLOCK:ETC2_R8G8B8A1_SRGB_BLOCK),
		(ETC2_R8G8B8A8_UNORM_BLOCK:ETC2_R8G8B8A8_SRGB_BLOCK),
		(EAC_R11_UNORM_BLOCK:EAC_R11_SNORM_BLOCK),
		(EAC_R11G11_UNORM_BLOCK:EAC_R11G11_SNORM_BLOCK),
		(ASTC_4X4_UNORM_BLOCK:ASTC_4X4_SRGB_BLOCK) || (ASTC_4X4_SFLOAT_BLOCK),
		(ASTC_5X4_UNORM_BLOCK:ASTC_5X4_SRGB_BLOCK) || (ASTC_5X4_SFLOAT_BLOCK),
		(ASTC_5X5_UNORM_BLOCK:ASTC_5X5_SRGB_BLOCK) || (ASTC_5X5_SFLOAT_BLOCK),
		(ASTC_6X5_UNORM_BLOCK:ASTC_6X5_SRGB_BLOCK) || (ASTC_6X5_SFLOAT_BLOCK),
		(ASTC_6X6_UNORM_BLOCK:ASTC_6X6_SRGB_BLOCK) || (ASTC_6X6_SFLOAT_BLOCK),
		(ASTC_8X5_UNORM_BLOCK:ASTC_8X5_SRGB_BLOCK) || (ASTC_8X5_SFLOAT_BLOCK),
		(ASTC_8X6_UNORM_BLOCK:ASTC_8X6_SRGB_BLOCK) || (ASTC_8X6_SFLOAT_BLOCK),
		(ASTC_8X8_UNORM_BLOCK:ASTC_8X8_SRGB_BLOCK) || (ASTC_8X8_SFLOAT_BLOCK),
		(ASTC_10X5_UNORM_BLOCK:ASTC_10X5_SRGB_BLOCK) || (ASTC_10X5_SFLOAT_BLOCK),
		(ASTC_10X6_UNORM_BLOCK:ASTC_10X6_SRGB_BLOCK) || (ASTC_10X6_SFLOAT_BLOCK),
		(ASTC_10X8_UNORM_BLOCK:ASTC_10X8_SRGB_BLOCK) || (ASTC_10X8_SFLOAT_BLOCK),
		(ASTC_10X10_UNORM_BLOCK:ASTC_10X10_SRGB_BLOCK) || (ASTC_10X10_SFLOAT_BLOCK),
		(ASTC_12X10_UNORM_BLOCK:ASTC_12X10_SRGB_BLOCK) || (ASTC_12X10_SFLOAT_BLOCK),
		(ASTC_12X12_UNORM_BLOCK:ASTC_12X12_SRGB_BLOCK) || (ASTC_12X12_SFLOAT_BLOCK),
	}
}
