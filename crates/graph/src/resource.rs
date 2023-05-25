use std::{hash::Hash, ops::BitOr, ptr::NonNull};

use ash::{
	vk,
	vk::{
		BorderColor,
		BufferCreateInfo,
		BufferUsageFlags,
		ComponentMapping,
		ComponentSwizzle,
		EventCreateFlags,
		EventCreateInfo,
		Extent3D,
		Filter,
		Format,
		ImageAspectFlags,
		ImageCreateFlags,
		ImageCreateInfo,
		ImageLayout,
		ImageMemoryRequirementsInfo2,
		ImageSubresourceRange,
		ImageType,
		ImageUsageFlags,
		ImageViewCreateInfo,
		ImageViewType,
		MemoryDedicatedRequirements,
		MemoryRequirements2,
		SampleCountFlags,
		SamplerAddressMode,
		SamplerCreateInfo,
		SamplerMipmapMode,
		SharingMode,
		REMAINING_ARRAY_LAYERS,
		REMAINING_MIP_LEVELS,
	},
};
use gpu_allocator::{
	vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
	MemoryLocation,
};

use crate::{
	device::{
		descriptor::{BufferId, ImageId, SamplerId, StorageImageId},
		Device,
		Queues,
	},
	Error,
	Result,
};

pub trait Resource: Default + Sized {
	type Desc: Eq + Hash + Copy;
	type Handle: Copy;

	fn handle(&self) -> Self::Handle;

	fn create(device: &Device, desc: Self::Desc) -> Result<Self>;

	unsafe fn destroy(self, device: &Device);
}

/// A description for a buffer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferDesc {
	pub size: usize,
	pub usage: BufferUsageFlags,
}

/// A buffer.
#[derive(Default)]
pub struct Buffer {
	inner: vk::Buffer,
	alloc: Allocation,
	id: Option<BufferId>,
}

impl Buffer {
	pub fn create(device: &Device, desc: BufferDesc, location: MemoryLocation) -> Result<Self> {
		let info = BufferCreateInfo::builder()
			.size(desc.size as u64)
			.usage(desc.usage)
			.sharing_mode(SharingMode::CONCURRENT);

		let usage = info.usage;
		let buffer = unsafe {
			match device.queue_families() {
				Queues::Single(q) => device.device().create_buffer(&info.queue_family_indices(&[q]), None),
				Queues::Separate {
					graphics,
					compute,
					transfer,
				} => device
					.device()
					.create_buffer(&info.queue_family_indices(&[graphics, compute, transfer]), None),
			}
		}?;

		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "Graph Buffer",
				requirements: unsafe { device.device().get_buffer_memory_requirements(buffer) },
				location,
				linear: true,
				allocation_scheme: AllocationScheme::GpuAllocatorManaged,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		unsafe {
			device
				.device()
				.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
		}

		let id = usage
			.contains(BufferUsageFlags::STORAGE_BUFFER)
			.then(|| device.base_descriptors().get_buffer(device.device(), buffer));

		Ok(Self {
			inner: buffer,
			alloc,
			id,
		})
	}

	pub fn size(&self) -> u64 { self.alloc.size() }

	pub fn mapped_ptr(&self) -> Option<NonNull<[u8]>> {
		unsafe {
			Some(NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
				self.alloc.mapped_ptr()?.as_ptr() as _,
				self.alloc.size() as _,
			)))
		}
	}

	pub fn inner(&self) -> vk::Buffer { self.inner }

	pub unsafe fn destroy(self, device: &Device) {
		if let Some(id) = self.id {
			device.base_descriptors().return_buffer(id);
		}

		let _ = device.allocator().free(self.alloc);
		device.device().destroy_buffer(self.inner, None);
	}
}

/// A handle to a buffer for uploading data from the CPU to the GPU.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct UploadBufferHandle {
	pub buffer: vk::Buffer,
	pub id: Option<BufferId>,
	pub data: NonNull<[u8]>,
}

/// A buffer for uploading data from the CPU to the GPU.
#[derive(Default)]
pub struct UploadBuffer {
	inner: Buffer,
}

impl Resource for UploadBuffer {
	type Desc = BufferDesc;
	type Handle = UploadBufferHandle;

	fn handle(&self) -> Self::Handle {
		UploadBufferHandle {
			buffer: self.inner.inner,
			data: self.inner.mapped_ptr().unwrap(),
			id: self.inner.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self>
	where
		Self: Sized,
	{
		Buffer::create(device, desc, MemoryLocation::CpuToGpu).map(|inner| Self { inner })
	}

	unsafe fn destroy(self, device: &Device) { self.inner.destroy(device) }
}

/// A handle to a buffer on the GPU.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct GpuBufferHandle {
	pub buffer: vk::Buffer,
	pub id: Option<BufferId>,
}

/// A buffer on the GPU.
#[derive(Default)]
pub struct GpuBuffer {
	inner: Buffer,
}

impl Resource for GpuBuffer {
	type Desc = BufferDesc;
	type Handle = GpuBufferHandle;

	fn handle(&self) -> Self::Handle {
		GpuBufferHandle {
			buffer: self.inner.inner,
			id: self.inner.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		Buffer::create(device, desc, MemoryLocation::GpuOnly).map(|inner| Self { inner })
	}

	unsafe fn destroy(self, device: &Device) { self.inner.destroy(device) }
}

/// A description for an image.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ImageDesc {
	pub flags: ImageCreateFlags,
	pub format: Format,
	pub size: Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: SampleCountFlags,
	pub usage: ImageUsageFlags,
}

/// A GPU-side image.
#[derive(Default)]
pub struct Image {
	inner: vk::Image,
	alloc: Allocation,
}

impl Resource for Image {
	type Desc = ImageDesc;
	type Handle = vk::Image;

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		let image = unsafe {
			device.device().create_image(
				&ImageCreateInfo::builder()
					.flags(desc.flags)
					.image_type(if desc.size.depth > 1 {
						ImageType::TYPE_3D
					} else if desc.size.height > 1 {
						ImageType::TYPE_2D
					} else {
						ImageType::TYPE_1D
					})
					.format(desc.format)
					.extent(desc.size)
					.mip_levels(desc.levels)
					.array_layers(desc.layers)
					.samples(desc.samples)
					.usage(desc.usage)
					.sharing_mode(SharingMode::EXCLUSIVE)
					.initial_layout(ImageLayout::UNDEFINED),
				None,
			)?
		};

		let (requirements, allocation_scheme) = unsafe {
			let mut dedicated = MemoryDedicatedRequirements::default();
			let mut out = MemoryRequirements2::builder().push_next(&mut dedicated);
			device
				.device()
				.get_image_memory_requirements2(&ImageMemoryRequirementsInfo2::builder().image(image), &mut out);

			(
				out.memory_requirements,
				match dedicated.prefers_dedicated_allocation != 0 || dedicated.requires_dedicated_allocation != 0 {
					true => AllocationScheme::DedicatedImage(image),
					false => AllocationScheme::GpuAllocatorManaged,
				},
			)
		};

		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "Graph Image",
				requirements,
				location: MemoryLocation::GpuOnly,
				linear: false,
				allocation_scheme,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		unsafe {
			device
				.device()
				.bind_image_memory(image, alloc.memory(), alloc.offset())?;
		}

		Ok(Self { inner: image, alloc })
	}

	unsafe fn destroy(self, device: &Device) {
		let _ = device.allocator().free(self.alloc);
		device.device().destroy_image(self.inner, None);
	}
}

/// The usage of an image view.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum ImageViewUsage {
	None,
	Sampled,
	Storage,
	Both,
}

impl BitOr for ImageViewUsage {
	type Output = Self;

	fn bitor(self, rhs: Self) -> Self::Output {
		match (self, rhs) {
			(ImageViewUsage::None, rhs) => rhs,
			(lhs, ImageViewUsage::None) => lhs,
			(ImageViewUsage::Both, _) | (_, ImageViewUsage::Both) => ImageViewUsage::Both,
			(ImageViewUsage::Sampled, ImageViewUsage::Storage) | (ImageViewUsage::Storage, ImageViewUsage::Sampled) => {
				ImageViewUsage::Both
			},
			(x @ ImageViewUsage::Sampled, ImageViewUsage::Sampled)
			| (x @ ImageViewUsage::Storage, ImageViewUsage::Storage) => x,
		}
	}
}

/// A description for an image view.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageViewDesc {
	pub image: vk::Image,
	pub view_type: ImageViewType,
	pub format: Format,
	pub usage: ImageViewUsage,
	pub aspect: ImageAspectFlags,
}

/// A GPU-side image view.
#[derive(Default, Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageView {
	pub image: vk::Image,
	pub view: vk::ImageView,
	pub id: Option<ImageId>,
	pub storage_id: Option<StorageImageId>,
}

impl Resource for ImageView {
	type Desc = ImageViewDesc;
	type Handle = Self;

	fn handle(&self) -> Self::Handle { *self }

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		unsafe {
			let view = device.device().create_image_view(
				&ImageViewCreateInfo::builder()
					.image(desc.image)
					.view_type(desc.view_type)
					.format(desc.format)
					.components(ComponentMapping {
						r: ComponentSwizzle::IDENTITY,
						g: ComponentSwizzle::IDENTITY,
						b: ComponentSwizzle::IDENTITY,
						a: ComponentSwizzle::IDENTITY,
					})
					.subresource_range(ImageSubresourceRange {
						aspect_mask: desc.aspect,
						base_mip_level: 0,
						level_count: REMAINING_MIP_LEVELS,
						base_array_layer: 0,
						layer_count: REMAINING_ARRAY_LAYERS,
					}),
				None,
			)?;
			let (id, storage_id) = match desc.usage {
				ImageViewUsage::None => (None, None),
				ImageViewUsage::Sampled => (Some(device.base_descriptors().get_image(device.device(), view)), None),
				ImageViewUsage::Storage => (
					None,
					Some(device.base_descriptors().get_storage_image(device.device(), view)),
				),
				ImageViewUsage::Both => (
					Some(device.base_descriptors().get_image(device.device(), view)),
					Some(device.base_descriptors().get_storage_image(device.device(), view)),
				),
			};

			Ok(Self {
				image: desc.image,
				view,
				id,
				storage_id,
			})
		}
	}

	unsafe fn destroy(self, device: &Device) {
		unsafe {
			if let Some(id) = self.id {
				device.base_descriptors().return_image(id);
			}
			if let Some(id) = self.storage_id {
				device.base_descriptors().return_storage_image(id);
			}
			device.device().destroy_image_view(self.view, None);
		}
	}
}

/// A description for a sampler.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct SamplerDesc {
	pub mag_filter: Filter,
	pub min_filter: Filter,
	pub mipmap_mode: SamplerMipmapMode,
	pub address_mode_u: SamplerAddressMode,
	pub address_mode_v: SamplerAddressMode,
	pub address_mode_w: SamplerAddressMode,
	pub border_color: BorderColor,
}

/// A GPU-side sampler.
#[derive(Default, Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct Sampler {
	pub sampler: vk::Sampler,
	pub id: Option<SamplerId>,
}

impl Resource for Sampler {
	type Desc = SamplerDesc;
	type Handle = Self;

	fn handle(&self) -> Self::Handle { *self }

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		unsafe {
			let sampler = device.device().create_sampler(
				&SamplerCreateInfo::builder()
					.mag_filter(desc.mag_filter)
					.min_filter(desc.min_filter)
					.mipmap_mode(desc.mipmap_mode)
					.address_mode_u(desc.address_mode_u)
					.address_mode_v(desc.address_mode_v)
					.address_mode_w(desc.address_mode_w)
					.border_color(desc.border_color),
				None,
			)?;
			let id = device.base_descriptors().get_sampler(device.device(), sampler);
			Ok(Self { sampler, id: Some(id) })
		}
	}

	unsafe fn destroy(self, device: &Device) {
		unsafe {
			device.base_descriptors().return_sampler(self.id.unwrap());
			device.device().destroy_sampler(self.sampler, None);
		}
	}
}

#[derive(Default)]
pub(crate) struct Event {
	inner: vk::Event,
}

impl Resource for Event {
	type Desc = ();
	type Handle = vk::Event;

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, _: Self::Desc) -> Result<Self> {
		unsafe {
			let inner = device
				.device()
				.create_event(&EventCreateInfo::builder().flags(EventCreateFlags::DEVICE_ONLY), None)?;
			Ok(Self { inner })
		}
	}

	unsafe fn destroy(self, device: &Device) { device.device().destroy_event(self.inner, None); }
}
