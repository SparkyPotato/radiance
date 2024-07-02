use std::{alloc::Allocator, collections::BTreeMap, hint::unreachable_unchecked, iter, ptr::NonNull};

use ash::vk;

pub use crate::sync::{BufferUsage as BufferUsageType, ImageUsage as ImageUsageType, Shader};
use crate::{
	arena::{Arena, IteratorAlloc, ToOwnedAlloc},
	device::{Device, QueueWait, QueueWaitOwned, SyncStage},
	graph::{compile::Resource, Caches, Res},
	resource::{BufferHandle, ImageView, ImageViewDescUnnamed, ImageViewUsage, Subresource},
	sync::UsageType,
};

/// A description for a GPU buffer.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferDesc {
	pub size: u64,
	pub upload: bool,
}

/// The usage of a buffer in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct BufferUsage<'a> {
	pub usages: &'a [BufferUsageType],
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferUsageOwned<A: Allocator> {
	pub usages: Vec<BufferUsageType, A>,
}

impl<A: Allocator> BufferUsageOwned<A> {
	fn default(alloc: A) -> Self {
		Self {
			usages: Vec::new_in(alloc),
		}
	}
}

impl<A: Allocator> ToOwnedAlloc<A> for BufferUsage<'_> {
	type Owned = BufferUsageOwned<A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
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
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
}

/// The usage of an image in a render pass.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ImageUsage<'a> {
	/// The format to view the image as. This can be different from the format in [`ImageDesc`], but must be
	/// [compatible].
	///
	/// [compatible]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#formats-compatibility-classes
	pub format: vk::Format,
	pub usages: &'a [ImageUsageType],
	pub view_type: vk::ImageViewType,
	pub subresource: Subresource,
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsageOwned<A: Allocator> {
	pub format: vk::Format,
	pub usages: Vec<ImageUsageType, A>,
	pub view_type: vk::ImageViewType,
	pub subresource: Subresource,
}

impl<A: Allocator> ImageUsageOwned<A> {
	fn default(alloc: A) -> Self {
		Self {
			format: vk::Format::UNDEFINED,
			usages: Vec::new_in(alloc),
			view_type: vk::ImageViewType::default(),
			subresource: Subresource::default(),
		}
	}
}

impl<A: Allocator> ToOwnedAlloc<A> for ImageUsage<'_> {
	type Owned = ImageUsageOwned<A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
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
			vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => vk::ImageCreateFlags::CUBE_COMPATIBLE,
			vk::ImageViewType::TYPE_2D_ARRAY => vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
			_ => vk::ImageCreateFlags::empty(),
		}
	}
}

/// Wait synchronization to be performed.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Wait<'a, U> {
	/// The related usage of the resource.
	pub usage: U,
	/// The cross-queue sync required (can be empty).
	pub wait: QueueWait<'a>,
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct WaitOwned<U, A: Allocator> {
	pub usage: U,
	pub wait: QueueWaitOwned<A>,
}

impl<U, A: Allocator> WaitOwned<U, A> {
	pub fn map<O>(self, f: impl FnOnce(U) -> O) -> WaitOwned<O, A> {
		WaitOwned {
			usage: f(self.usage),
			wait: self.wait,
		}
	}
}

impl<A: Allocator + Clone, U: ToOwnedAlloc<A>> ToOwnedAlloc<A> for Wait<'_, U> {
	type Owned = WaitOwned<U::Owned, A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
		Self::Owned {
			usage: self.usage.to_owned_alloc(alloc.clone()),
			wait: self.wait.to_owned_alloc(alloc),
		}
	}
}

/// Signal synchronization to be performed.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Signal<'a, U> {
	/// The related usage of the resource.
	pub usage: U,
	/// The cross-queue sync required (can be empty).
	pub signal: &'a [SyncStage<vk::Semaphore>],
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SignalOwned<U, A: Allocator> {
	pub usage: U,
	pub signal: Vec<SyncStage<vk::Semaphore>, A>,
}

impl<U, A: Allocator> SignalOwned<U, A> {
	pub fn map<O>(self, f: impl FnOnce(U) -> O) -> SignalOwned<O, A> {
		SignalOwned {
			usage: f(self.usage),
			signal: self.signal,
		}
	}
}

impl<A: Allocator + Clone, U: ToOwnedAlloc<A>> ToOwnedAlloc<A> for Signal<'_, U> {
	type Owned = SignalOwned<U::Owned, A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
		Self::Owned {
			usage: self.usage.to_owned_alloc(alloc.clone()),
			signal: self.signal.to_owned_alloc(alloc),
		}
	}
}

/// A buffer external to the render graph.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBuffer<'a> {
	/// The handle to the buffer. This is passed as-is to the render pass.
	pub handle: BufferHandle,
	/// The external usage of the buffer before the render pass is executed.
	pub wait: Wait<'a, BufferUsage<'a>>,
	/// The external usage of the buffer after the render pass is executed. This is usually not required.
	pub signal: Signal<'a, BufferUsage<'a>>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBufferOwned<A: Allocator> {
	pub handle: BufferHandle,
	pub prev_usage: WaitOwned<BufferUsageOwned<A>, A>,
	pub next_usage: SignalOwned<BufferUsageOwned<A>, A>,
}

impl<A: Allocator + Clone> ToOwnedAlloc<A> for ExternalBuffer<'_> {
	type Owned = ExternalBufferOwned<A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
		Self::Owned {
			handle: self.handle,
			prev_usage: self.wait.to_owned_alloc(alloc.clone()),
			next_usage: self.signal.to_owned_alloc(alloc),
		}
	}
}

/// An image external to the render graph.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalImage<'a> {
	pub handle: vk::Image,
	pub size: vk::Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
	/// The external usage of the image before the render pass is executed.
	pub wait: Wait<'a, ImageUsage<'a>>,
	/// The external usage of the image after the render pass is executed. This is usually not required.
	pub signal: Signal<'a, ImageUsage<'a>>,
}

pub trait Usage {
	type Inner: Copy + Into<UsageType>;

	fn get_usages(&self) -> &[Self::Inner];
}

impl<A: Allocator> Usage for BufferUsageOwned<A> {
	type Inner = BufferUsageType;

	fn get_usages(&self) -> &[Self::Inner] { &self.usages }
}

impl<A: Allocator> Usage for ImageUsageOwned<A> {
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
		self, pass: u32, write_usage: <Self::Resource as VirtualResource>::Usage<'_>,
		resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>, base_id: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph>;
}

pub trait VirtualResource {
	type Usage<'a>;

	unsafe fn from_res(pass: u32, res: &mut Resource, caches: &mut Caches, device: &Device) -> Self;

	unsafe fn add_read_usage(ty: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>);
}

#[derive(Clone)]
pub struct GpuData<'graph, D, H, U> {
	pub desc: D,
	pub handle: H,
	pub usages: BTreeMap<u32, U, &'graph Arena>,
	pub wait: WaitOwned<U, &'graph Arena>,
	pub signal: SignalOwned<U, &'graph Arena>,
}

pub type BufferData<'graph> = GpuData<'graph, BufferDesc, BufferHandle, BufferUsageOwned<&'graph Arena>>;
pub type ImageData<'graph> = GpuData<'graph, ImageDesc, vk::Image, ImageUsageOwned<&'graph Arena>>;

#[derive(Clone)]
pub enum VirtualResourceType<'graph> {
	Data(NonNull<()>),
	Buffer(BufferData<'graph>),
	Image(ImageData<'graph>),
}

impl<'graph> VirtualResourceType<'graph> {
	unsafe fn buffer(&mut self) -> &mut BufferData<'graph> {
		match self {
			VirtualResourceType::Buffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&mut self) -> &mut ImageData<'graph> {
		match self {
			VirtualResourceType::Image(data) => data,
			_ => unreachable_unchecked(),
		}
	}
}

impl VirtualResource for BufferHandle {
	type Usage<'a> = BufferUsage<'a>;

	unsafe fn from_res(_: u32, res: &mut Resource, _: &mut Caches, _: &Device) -> Self { res.buffer().handle }

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>) {
		let b = res.ty.buffer();
		b.usages.insert(pass, usage.to_owned_alloc(b.signal.signal.allocator()));
	}
}

impl VirtualResourceDesc for BufferDesc {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsage<'_>, _: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		_: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Buffer(BufferData {
			desc: self,
			handle: BufferHandle::default(),
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: WaitOwned {
				usage: BufferUsageOwned::default(arena),
				wait: QueueWaitOwned::default(arena),
			},
			signal: SignalOwned {
				usage: BufferUsageOwned::default(arena),
				signal: Vec::new_in(arena),
			},
		})
	}
}

impl VirtualResource for ImageView {
	type Usage<'a> = ImageUsage<'a>;

	unsafe fn from_res(pass: u32, res: &mut Resource, caches: &mut Caches, device: &Device) -> Self {
		let image = res.image();
		let usage = &image.usages[&pass];

		caches
			.image_views
			.get(
				device,
				ImageViewDescUnnamed {
					image: image.handle,
					size: image.desc.size,
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
					subresource: usage.subresource,
				},
			)
			.expect("Failed to create image view")
	}

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage<'_>) {
		let image = res.ty.image();
		let fmt = image.usages.first_key_value().unwrap().1.format;
		debug_assert!(
			compatible_formats(fmt, usage.format),
			"`{:?}` and `{:?}` are not compatible",
			fmt,
			usage.format
		);
		image
			.usages
			.insert(pass, usage.to_owned_alloc(image.signal.signal.allocator()));
	}
}

impl VirtualResourceDesc for ImageDesc {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsage<'_>, _: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		_: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(ImageData {
			desc: self,
			handle: Default::default(),
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: WaitOwned {
				usage: ImageUsageOwned::default(arena),
				wait: QueueWaitOwned::default(arena),
			},
			signal: SignalOwned {
				usage: ImageUsageOwned::default(arena),
				signal: Vec::new_in(arena),
			},
		})
	}
}

impl VirtualResourceDesc for ExternalBuffer<'_> {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsage<'_>, _: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		_: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Buffer(BufferData {
			desc: BufferDesc {
				size: self.handle.data.len() as _,
				upload: false,
			},
			handle: self.handle,
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: self.wait.to_owned_alloc(arena),
			signal: self.signal.to_owned_alloc(arena),
		})
	}
}

impl VirtualResourceDesc for ExternalImage<'_> {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsage<'_>, _: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		_: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(ImageData {
			desc: ImageDesc {
				size: self.size,
				levels: self.levels,
				layers: self.layers,
				samples: self.samples,
			},
			handle: self.handle,
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: self.wait.to_owned_alloc(arena),
			signal: self.signal.to_owned_alloc(arena),
		})
	}
}

impl VirtualResourceDesc for Res<BufferHandle> {
	type Resource = BufferHandle;

	fn ty<'graph>(
		self, pass: u32, write_usage: BufferUsage<'_>, resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		base_id: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Buffer(BufferData {
			desc: unsafe { resources[self.id - base_id].ty.buffer().desc.clone() },
			handle: BufferHandle::default(),
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: WaitOwned {
				usage: BufferUsageOwned::default(arena),
				wait: QueueWaitOwned::default(arena),
			},
			signal: SignalOwned {
				usage: BufferUsageOwned::default(arena),
				signal: Vec::new_in(arena),
			},
		})
	}
}

impl VirtualResourceDesc for Res<ImageView> {
	type Resource = ImageView;

	fn ty<'graph>(
		self, pass: u32, write_usage: ImageUsage<'_>, resources: &mut Vec<VirtualResourceData<'graph>, &'graph Arena>,
		base_id: usize, arena: &'graph Arena,
	) -> VirtualResourceType<'graph> {
		VirtualResourceType::Image(ImageData {
			desc: unsafe { resources[self.id - base_id].ty.image().desc.clone() },
			handle: Default::default(),
			usages: iter::once((pass, write_usage.to_owned_alloc(arena))).collect_in(arena),
			wait: WaitOwned {
				usage: ImageUsageOwned::default(arena),
				wait: QueueWaitOwned::default(arena),
			},
			signal: SignalOwned {
				usage: ImageUsageOwned::default(arena),
				signal: Vec::new_in(arena),
			},
		})
	}
}

pub fn compatible_formats(a: vk::Format, b: vk::Format) -> bool {
	// get_format_block(a) == get_format_block(b)  TODO: fix
	a == b
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
