use std::{alloc::Allocator, collections::BTreeMap, hint::unreachable_unchecked, iter, ptr::NonNull};

use ash::vk;

pub use crate::sync::{BufferUsage as BufferUsageType, ImageUsage as ImageUsageType, Shader};
use crate::{
	arena::{Arena, IteratorAlloc, ToOwnedAlloc},
	device::Device,
	graph::{cache::Persist, compile::Resource, Caches, Res},
	resource::{
		Buffer,
		BufferHandle,
		Image,
		ImageView,
		ImageViewDescUnnamed,
		ImageViewUsage,
		Resource as _,
		Subresource,
	},
};

/// The location of a GPU buffer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum BufferLoc {
	Upload,
	Staging,
	Gpu,
	Readback,
}

/// A description for a GPU buffer.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferDesc {
	pub size: u64,
	pub loc: BufferLoc,
	pub persist: Option<Persist<Buffer>>,
}

impl BufferDesc {
	pub fn upload(size: u64) -> Self {
		Self {
			size,
			loc: BufferLoc::Upload,
			persist: None,
		}
	}

	pub fn staging(size: u64) -> Self {
		Self {
			size,
			loc: BufferLoc::Staging,
			persist: None,
		}
	}

	pub fn gpu(size: u64) -> Self {
		Self {
			size,
			loc: BufferLoc::Gpu,
			persist: None,
		}
	}

	pub fn persist(self, persist: Persist<Buffer>) -> Self {
		Self {
			persist: Some(persist),
			..self
		}
	}

	pub fn readback(size: u64, persist: Persist<Buffer>) -> Self {
		Self {
			size,
			loc: BufferLoc::Readback,
			persist: Some(persist),
		}
	}
}

/// The usage of a buffer in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct BufferUsage<'a> {
	pub usages: &'a [BufferUsageType],
}

#[doc(hidden)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferUsageArray<const N: usize> {
	pub usages: [BufferUsageType; N],
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferUsageOwned<A: Allocator> {
	pub usages: Vec<BufferUsageType, A>,
}

impl BufferUsage<'_> {
	pub fn none() -> BufferUsageArray<0> { BufferUsageArray { usages: [] } }

	pub fn read(shader: Shader) -> BufferUsageArray<1> {
		BufferUsageArray {
			usages: [BufferUsageType::ShaderStorageRead(shader)],
		}
	}

	pub fn write(shader: Shader) -> BufferUsageArray<1> {
		BufferUsageArray {
			usages: [BufferUsageType::ShaderStorageWrite(shader)],
		}
	}

	pub fn read_write(shader: Shader) -> BufferUsageArray<2> {
		BufferUsageArray {
			usages: [
				BufferUsageType::ShaderStorageRead(shader),
				BufferUsageType::ShaderStorageWrite(shader),
			],
		}
	}

	pub fn index() -> BufferUsageArray<1> {
		BufferUsageArray {
			usages: [BufferUsageType::IndexBuffer],
		}
	}

	pub fn transfer_read() -> BufferUsageArray<1> {
		BufferUsageArray {
			usages: [BufferUsageType::TransferRead],
		}
	}

	pub fn transfer_write() -> BufferUsageArray<1> {
		BufferUsageArray {
			usages: [BufferUsageType::TransferWrite],
		}
	}
}

impl ToOwnedAlloc for BufferUsage<'_> {
	type Owned<A: Allocator> = BufferUsageOwned<A>;

	fn to_owned_alloc<A: Allocator>(&self, alloc: A) -> Self::Owned<A> {
		Self::Owned {
			usages: self.usages.to_owned_alloc(alloc),
		}
	}
}

impl<const N: usize> ToOwnedAlloc for BufferUsageArray<N> {
	type Owned<A: Allocator> = BufferUsageOwned<A>;

	fn to_owned_alloc<A: Allocator>(&self, alloc: A) -> Self::Owned<A> {
		Self::Owned {
			usages: self.usages.to_owned_alloc(alloc),
		}
	}
}

/// A description for an image.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageDesc {
	pub size: vk::Extent3D,
	pub format: vk::Format,
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
	pub persist: Option<Persist<Image>>,
}

impl Default for ImageDesc {
	fn default() -> Self {
		Self {
			size: vk::Extent3D {
				width: 1,
				height: 1,
				depth: 1,
			},
			format: vk::Format::UNDEFINED,
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
			persist: None,
		}
	}
}

/// The usage of an image in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ImageUsage<'a> {
	/// The format to view the image as. This can be different from the format in [`ImageDesc`], but must be
	/// [compatible].
	///
	/// Can be undefined, in which case the creation-time format of the image is used.
	///
	/// [compatible]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#formats-compatibility-classes
	pub format: vk::Format,
	pub usages: &'a [ImageUsageType],
	pub view_type: Option<vk::ImageViewType>,
	pub subresource: Subresource,
}

#[doc(hidden)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsageArray<const N: usize> {
	pub format: vk::Format,
	pub usages: [ImageUsageType; N],
	pub view_type: Option<vk::ImageViewType>,
	pub subresource: Subresource,
}

impl ImageUsage<'_> {
	pub fn d2<const N: usize>(format: vk::Format, usages: [ImageUsageType; N]) -> ImageUsageArray<N> {
		ImageUsageArray {
			format,
			usages,
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		}
	}

	pub fn sampled_2d(shader: Shader) -> ImageUsageArray<1> { Self::format_sampled_2d(vk::Format::UNDEFINED, shader) }

	pub fn format_sampled_2d(format: vk::Format, shader: Shader) -> ImageUsageArray<1> {
		Self::d2(format, [ImageUsageType::ShaderReadSampledImage(shader)])
	}

	pub fn read_2d(shader: Shader) -> ImageUsageArray<1> {
		Self::d2(vk::Format::UNDEFINED, [ImageUsageType::ShaderStorageRead(shader)])
	}

	pub fn write_2d(shader: Shader) -> ImageUsageArray<1> {
		Self::d2(vk::Format::UNDEFINED, [ImageUsageType::ShaderStorageWrite(shader)])
	}

	pub fn read_write_2d(shader: Shader) -> ImageUsageArray<2> {
		Self::d2(
			vk::Format::UNDEFINED,
			[
				ImageUsageType::ShaderStorageRead(shader),
				ImageUsageType::ShaderStorageWrite(shader),
			],
		)
	}

	pub fn color_attachment() -> ImageUsageArray<1> { Self::format_color_attachment(vk::Format::UNDEFINED) }

	pub fn format_color_attachment(format: vk::Format) -> ImageUsageArray<1> {
		Self::d2(format, [ImageUsageType::ColorAttachmentWrite])
	}

	pub fn no_view<const N: usize>(usages: [ImageUsageType; N]) -> ImageUsageArray<N> {
		ImageUsageArray {
			format: vk::Format::UNDEFINED,
			usages,
			view_type: None,
			subresource: Subresource::default(),
		}
	}

	pub fn transfer_read() -> ImageUsageArray<1> { Self::no_view([ImageUsageType::TransferRead]) }

	pub fn transfer_write() -> ImageUsageArray<1> { Self::no_view([ImageUsageType::TransferWrite]) }
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsageOwned<A: Allocator> {
	pub format: vk::Format,
	pub usages: Vec<ImageUsageType, A>,
	pub view_type: Option<vk::ImageViewType>,
	pub subresource: Subresource,
}

impl ToOwnedAlloc for ImageUsage<'_> {
	type Owned<A: Allocator> = ImageUsageOwned<A>;

	fn to_owned_alloc<A: Allocator>(&self, alloc: A) -> Self::Owned<A> {
		Self::Owned {
			format: self.format,
			usages: self.usages.to_owned_alloc(alloc),
			view_type: self.view_type,
			subresource: self.subresource,
		}
	}
}

impl<const N: usize> ToOwnedAlloc for ImageUsageArray<N> {
	type Owned<A: Allocator> = ImageUsageOwned<A>;

	fn to_owned_alloc<A: Allocator>(&self, alloc: A) -> Self::Owned<A> {
		Self::Owned {
			format: self.format,
			usages: self.usages.to_owned_alloc(alloc),
			view_type: self.view_type,
			subresource: self.subresource,
		}
	}
}

impl<A: Allocator> ImageUsageOwned<A> {
	pub fn create_flags(&self) -> vk::ImageCreateFlags {
		match self.view_type {
			Some(vk::ImageViewType::CUBE) | Some(vk::ImageViewType::CUBE_ARRAY) => {
				vk::ImageCreateFlags::CUBE_COMPATIBLE
			},
			Some(vk::ImageViewType::TYPE_2D_ARRAY) => vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
			_ => vk::ImageCreateFlags::empty(),
		}
	}
}

/// A buffer external to the render graph.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBuffer {
	/// The handle to the buffer. This is passed as-is to the render pass.
	pub handle: BufferHandle,
}

impl ExternalBuffer {
	pub fn new(buf: &Buffer) -> Self { Self { handle: buf.handle() } }
}

/// An image external to the render graph.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalImage {
	pub handle: vk::Image,
	pub layout: vk::ImageLayout,
	pub desc: ImageDesc,
}

/// A swapchain image imported into the render graph.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct SwapchainImage {
	pub handle: vk::Image,
	pub size: vk::Extent2D,
	pub format: vk::Format,
	pub available: vk::Semaphore,
	pub rendered: vk::Semaphore,
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
		self, pass: u32, write_usage: <Self::Resource as VirtualResource>::Usage<&'graph Arena>,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph>;
}

pub trait VirtualResource {
	type Usage<A: Allocator>;
	type Desc;

	unsafe fn desc(ty: &VirtualResourceData) -> Self::Desc;

	unsafe fn from_res(pass: u32, res: &mut Resource, caches: &mut Caches, device: &Device) -> Self;

	unsafe fn add_read_usage<'a>(ty: &mut VirtualResourceData<'a>, pass: u32, usage: Self::Usage<&'a Arena>);
}

#[derive(Clone)]
pub struct GpuData<'graph, D, H, U> {
	pub desc: D,
	pub handle: H,
	pub uninit: bool,
	pub usages: BTreeMap<u32, U, &'graph Arena>,
	pub swapchain: Option<(vk::Semaphore, vk::Semaphore)>,
}

pub type BufferData<'graph> = GpuData<'graph, BufferDesc, BufferHandle, BufferUsageOwned<&'graph Arena>>;
pub type ImageData<'graph> = GpuData<'graph, ImageDesc, (vk::Image, vk::ImageLayout), ImageUsageOwned<&'graph Arena>>;

#[derive(Clone)]
pub enum VirtualResourceType<'graph> {
	Data(NonNull<()>),
	Buffer(BufferData<'graph>),
	Image(ImageData<'graph>),
}

impl<'graph> VirtualResourceType<'graph> {
	unsafe fn buffer_mut(&mut self) -> &mut BufferData<'graph> {
		match self {
			VirtualResourceType::Buffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn buffer(&self) -> &BufferData<'graph> {
		match self {
			VirtualResourceType::Buffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image_mut(&mut self) -> &mut ImageData<'graph> {
		match self {
			VirtualResourceType::Image(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&self) -> &ImageData<'graph> {
		match self {
			VirtualResourceType::Image(data) => data,
			_ => unreachable_unchecked(),
		}
	}
}

impl VirtualResource for BufferHandle {
	type Desc = BufferDesc;
	type Usage<A: Allocator> = BufferUsageOwned<A>;

	unsafe fn desc(ty: &VirtualResourceData) -> Self::Desc {
		let mut d = ty.ty.buffer().desc;
		d.persist = None;
		d
	}

	unsafe fn from_res(_: u32, res: &mut Resource, _: &mut Caches, _: &Device) -> Self { res.buffer().handle }

	unsafe fn add_read_usage<'a>(res: &mut VirtualResourceData<'a>, pass: u32, usage: BufferUsageOwned<&'a Arena>) {
		let b = res.ty.buffer_mut();
		b.usages.insert(pass, usage);
	}
}

impl VirtualResourceDesc for BufferDesc {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsageOwned<&'graph Arena>,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Buffer(BufferData {
			desc: self,
			handle: BufferHandle::default(),
			uninit: true,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

impl VirtualResource for ImageView {
	type Desc = ImageDesc;
	type Usage<A: Allocator> = ImageUsageOwned<A>;

	unsafe fn desc(ty: &VirtualResourceData) -> Self::Desc {
		let mut d = ty.ty.image().desc;
		d.persist = None;
		d
	}

	unsafe fn from_res(pass: u32, res: &mut Resource, caches: &mut Caches, device: &Device) -> Self {
		let image = res.image();
		let usage = image.usages.get(&pass).expect("Resource was never referenced in pass");

		if let Some(view_type) = usage.view_type {
			caches
				.image_views
				.get(
					device,
					ImageViewDescUnnamed {
						image: image.handle.0,
						size: image.desc.size,
						view_type,
						format: if usage.format == vk::Format::UNDEFINED {
							image.desc.format
						} else {
							usage.format
						},
						usage: usage.usages.iter().fold(ImageViewUsage::None, |u, i| {
							u | match i {
								ImageUsageType::ShaderStorageRead(_) | ImageUsageType::ShaderStorageWrite(_) => {
									ImageViewUsage::Storage
								},
								ImageUsageType::ShaderReadSampledImage(_) => ImageViewUsage::Sampled,
								ImageUsageType::General => ImageViewUsage::Both,
								_ => ImageViewUsage::None,
							}
						}),
						subresource: usage.subresource,
					},
				)
				.expect("Failed to create image view")
				.0
		} else {
			ImageView {
				image: image.handle.0,
				view: vk::ImageView::null(),
				id: None,
				storage_id: None,
				size: image.desc.size,
			}
		}
	}

	unsafe fn add_read_usage<'a>(res: &mut VirtualResourceData<'a>, pass: u32, usage: ImageUsageOwned<&'a Arena>) {
		let image = res.ty.image_mut();
		debug_assert!(
			compatible_formats(image.desc.format, usage.format),
			"`{:?}` and `{:?}` are not compatible",
			image.desc.format,
			usage.format
		);
		image.usages.insert(pass, usage);
	}
}

impl VirtualResourceDesc for ImageDesc {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsageOwned<&'graph Arena>,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Image(ImageData {
			desc: self,
			handle: Default::default(),
			uninit: true,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

impl VirtualResourceDesc for ExternalBuffer {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsageOwned<&'graph Arena>,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Buffer(BufferData {
			desc: BufferDesc {
				size: self.handle.data.len() as _,
				loc: BufferLoc::Gpu,
				persist: None,
			},
			handle: self.handle,
			uninit: false,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

impl VirtualResourceDesc for ExternalImage {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsageOwned<&'graph Arena>,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		assert!(
			compatible_formats(self.desc.format, write_usage.format),
			"External image format is invalid: is {:?}, pass expected {:?}",
			self.desc.format,
			write_usage.format,
		);
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Image(ImageData {
			desc: self.desc,
			handle: (self.handle, self.layout),
			uninit: false,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

impl VirtualResourceDesc for SwapchainImage {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsageOwned<&'graph Arena>,
		_: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, _: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Image(ImageData {
			desc: ImageDesc {
				size: vk::Extent3D {
					width: self.size.width,
					height: self.size.height,
					depth: 1,
				},
				format: self.format,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				persist: None,
			},
			handle: (self.handle, vk::ImageLayout::UNDEFINED),
			uninit: false,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: Some((self.available, self.rendered)),
		})
	}
}

impl VirtualResourceDesc for Res<BufferHandle> {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsageOwned<&'graph Arena>,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Buffer(BufferData {
			desc: unsafe { resources[self.id - base_id].ty.buffer_mut().desc.clone() },
			handle: BufferHandle::default(),
			uninit: true,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

impl VirtualResourceDesc for Res<ImageView> {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsageOwned<&'graph Arena>,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize,
	) -> VirtualResourceType<'graph> {
		let arena = *write_usage.usages.allocator();
		VirtualResourceType::Image(ImageData {
			desc: unsafe { resources[self.id - base_id].ty.image_mut().desc.clone() },
			handle: Default::default(),
			uninit: true,
			usages: iter::once((pass, write_usage)).collect_in(arena),
			swapchain: None,
		})
	}
}

pub fn compatible_formats(a: vk::Format, b: vk::Format) -> bool {
	get_format_block(a) == get_format_block(b) || a == vk::Format::UNDEFINED || b == vk::Format::UNDEFINED
}

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
