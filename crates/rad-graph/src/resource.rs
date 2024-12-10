use std::{ffi::CString, hash::Hash, marker::PhantomData, ops::BitOr, ptr::NonNull};

use ash::vk;
use bytemuck::{NoUninit, Pod, Zeroable};
use gpu_allocator::{
	vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
	MemoryLocation,
};

use crate::{
	device::{
		descriptor::{ImageId, StorageImageId},
		Device,
		Queues,
	},
	graph,
	Error,
	Result,
};

pub trait ToNamed {
	type Named<'a>: Named<'a>;

	fn to_named(self, name: &str) -> Self::Named<'_>;
}

pub trait Named<'a> {
	fn name(&self) -> &'a str;
}

pub trait Resource: Default + Sized {
	type Desc<'a>: Copy + Eq + Hash + Named<'a>;
	type UnnamedDesc: Copy + Eq + Hash + for<'a> ToNamed<Named<'a> = Self::Desc<'a>>;
	type Handle: Copy;

	fn handle(&self) -> Self::Handle;

	fn create(device: &Device, desc: Self::Desc<'_>) -> Result<Self>;

	/// # Safety
	/// The resource must not be used after being destroyed, and appropriate synchronization must be performed.
	unsafe fn destroy(self, device: &Device);
}

/// A description for a buffer.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct BufferDesc<'a> {
	pub name: &'a str,
	pub size: u64,
	pub readback: bool,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferDescUnnamed {
	pub size: u64,
	pub readback: bool,
}

impl<'a> Named<'a> for BufferDesc<'a> {
	fn name(&self) -> &'a str { self.name }
}

impl ToNamed for BufferDescUnnamed {
	type Named<'a> = BufferDesc<'a>;

	fn to_named(self, name: &str) -> Self::Named<'_> {
		Self::Named {
			name,
			size: self.size,
			readback: self.readback,
		}
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
pub struct GpuPtr<T: NoUninit>(pub u64, PhantomData<fn() -> T>);
unsafe impl<T: NoUninit> Zeroable for GpuPtr<T> {}
unsafe impl<T: NoUninit> Pod for GpuPtr<T> {}
impl<T: NoUninit> GpuPtr<T> {
	pub fn null() -> Self { Self(0, PhantomData) }

	pub fn addr(&self) -> u64 { self.0 }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferHandle {
	pub buffer: vk::Buffer,
	pub addr: u64,
	pub data: NonNull<[u8]>,
}

impl Default for BufferHandle {
	fn default() -> Self {
		Self {
			buffer: vk::Buffer::null(),
			addr: 0,
			data: unsafe { NonNull::new_unchecked(std::slice::from_raw_parts_mut(std::ptr::dangling_mut(), 0)) },
		}
	}
}

impl BufferHandle {
	pub fn ptr<T: NoUninit>(&self) -> GpuPtr<T> { GpuPtr(self.addr, PhantomData) }
}

/// A buffer.
#[derive(Default)]
pub struct Buffer {
	inner: vk::Buffer,
	alloc: Allocation,
	addr: u64,
}

impl Buffer {
	pub fn size(&self) -> u64 { self.alloc.size() }

	pub fn data(&self) -> NonNull<[u8]> {
		unsafe {
			NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
				self.alloc.mapped_ptr().unwrap().as_ptr() as _,
				self.alloc.size() as _,
			))
		}
	}

	pub fn inner(&self) -> vk::Buffer { self.inner }

	pub fn ptr<T: NoUninit>(&self) -> GpuPtr<T> { GpuPtr(self.addr, PhantomData) }

	pub fn desc(&self) -> graph::BufferDesc {
		graph::BufferDesc {
			size: self.data().len() as _,
			loc: if self
				.alloc
				.memory_properties()
				.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
			{
				graph::BufferLoc::Readback
			} else {
				graph::BufferLoc::GpuOnly
			},
			persist: None,
		}
	}
}

impl Resource for Buffer {
	type Desc<'a> = BufferDesc<'a>;
	type Handle = BufferHandle;
	type UnnamedDesc = BufferDescUnnamed;

	fn handle(&self) -> Self::Handle {
		BufferHandle {
			buffer: self.inner,
			addr: self.addr,
			data: self.data(),
		}
	}

	fn create(device: &Device, desc: Self::Desc<'_>) -> Result<Self>
	where
		Self: Sized,
	{
		unsafe {
			let info = vk::BufferCreateInfo::default()
				.size(desc.size)
				.usage(
					vk::BufferUsageFlags::TRANSFER_SRC
						| vk::BufferUsageFlags::TRANSFER_DST
						| vk::BufferUsageFlags::STORAGE_BUFFER
						| vk::BufferUsageFlags::INDEX_BUFFER
						| vk::BufferUsageFlags::VERTEX_BUFFER
						| vk::BufferUsageFlags::INDIRECT_BUFFER
						| vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
						| vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
						| vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
						| vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
				)
				.sharing_mode(vk::SharingMode::CONCURRENT);

			let Queues {
				graphics,
				compute,
				transfer,
			} = device.queue_families();
			let buffer = device
				.device()
				.create_buffer(&info.queue_family_indices(&[graphics, compute, transfer]), None)?;

			let _ = device.debug_utils_ext().map(|d| {
				let name = CString::new(desc.name).unwrap();
				d.set_debug_utils_object_name(
					&vk::DebugUtilsObjectNameInfoEXT::default()
						.object_handle(buffer)
						.object_name(&name),
				)
			});

			let alloc = device
				.allocator()
				.allocate(&AllocationCreateDesc {
					name: desc.name,
					requirements: device.device().get_buffer_memory_requirements(buffer),
					location: MemoryLocation::CpuToGpu,
					linear: true,
					allocation_scheme: AllocationScheme::GpuAllocatorManaged,
				})
				.map_err(|e| Error::Message(e.to_string()))?;

			device
				.device()
				.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;

			let addr = device
				.device()
				.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

			Ok(Self {
				inner: buffer,
				alloc,
				addr,
			})
		}
	}

	unsafe fn destroy(self, device: &Device) {
		let _ = device.allocator().free(self.alloc);
		device.device().destroy_buffer(self.inner, None);
	}
}

/// A description for an image.
#[derive(Copy, Clone, Default, Hash, PartialEq, Eq)]
pub struct ImageDesc<'a> {
	pub name: &'a str,
	pub flags: vk::ImageCreateFlags,
	pub format: vk::Format,
	pub size: vk::Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
	pub usage: vk::ImageUsageFlags,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ImageDescUnnamed {
	pub flags: vk::ImageCreateFlags,
	pub format: vk::Format,
	pub size: vk::Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: vk::SampleCountFlags,
	pub usage: vk::ImageUsageFlags,
}

impl<'a> Named<'a> for ImageDesc<'a> {
	fn name(&self) -> &'a str { self.name }
}

impl ToNamed for ImageDescUnnamed {
	type Named<'a> = ImageDesc<'a>;

	fn to_named(self, name: &str) -> Self::Named<'_> {
		Self::Named {
			name,
			flags: self.flags,
			format: self.format,
			size: self.size,
			levels: self.levels,
			layers: self.layers,
			samples: self.samples,
			usage: self.usage,
		}
	}
}

/// A GPU-side image.
#[derive(Default)]
pub struct Image {
	inner: vk::Image,
	desc: graph::ImageDesc,
	alloc: Allocation,
}

impl Image {
	pub fn desc(&self) -> graph::ImageDesc { self.desc }
}

impl Resource for Image {
	type Desc<'a> = ImageDesc<'a>;
	type Handle = vk::Image;
	type UnnamedDesc = ImageDescUnnamed;

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, desc: Self::Desc<'_>) -> Result<Self> {
		unsafe {
			let Queues {
				graphics,
				compute,
				transfer,
			} = device.queue_families();
			let image = device.device().create_image(
				&vk::ImageCreateInfo::default()
					.flags(desc.flags)
					.image_type(if desc.size.depth > 1 {
						vk::ImageType::TYPE_3D
					} else if desc.size.height > 1 {
						vk::ImageType::TYPE_2D
					} else {
						vk::ImageType::TYPE_1D
					})
					.format(desc.format)
					.extent(desc.size)
					.mip_levels(desc.levels)
					.array_layers(desc.layers)
					.samples(desc.samples)
					.usage(desc.usage)
					.sharing_mode(vk::SharingMode::CONCURRENT)
					.queue_family_indices(&[graphics, compute, transfer])
					.initial_layout(vk::ImageLayout::UNDEFINED),
				None,
			)?;
			let _ = device.debug_utils_ext().map(|d| {
				let name = CString::new(desc.name).unwrap();
				d.set_debug_utils_object_name(
					&vk::DebugUtilsObjectNameInfoEXT::default()
						.object_handle(image)
						.object_name(&name),
				)
			});

			let mut dedicated = vk::MemoryDedicatedRequirements::default();
			let mut out = vk::MemoryRequirements2::default().push_next(&mut dedicated);
			device
				.device()
				.get_image_memory_requirements2(&vk::ImageMemoryRequirementsInfo2::default().image(image), &mut out);

			let alloc = device
				.allocator()
				.allocate(&AllocationCreateDesc {
					name: desc.name,
					requirements: out.memory_requirements,
					location: MemoryLocation::GpuOnly,
					linear: false,
					allocation_scheme: match dedicated.prefers_dedicated_allocation != 0
						|| dedicated.requires_dedicated_allocation != 0
					{
						true => AllocationScheme::DedicatedImage(image),
						false => AllocationScheme::GpuAllocatorManaged,
					},
				})
				.map_err(|e| Error::Message(e.to_string()))?;

			device
				.device()
				.bind_image_memory(image, alloc.memory(), alloc.offset())?;

			Ok(Self {
				inner: image,
				alloc,
				desc: graph::ImageDesc {
					size: desc.size,
					levels: desc.levels,
					layers: desc.layers,
					format: desc.format,
					samples: desc.samples,
					persist: None,
				},
			})
		}
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

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct Subresource {
	pub aspect: vk::ImageAspectFlags,
	pub first_layer: u32,
	pub layer_count: u32,
	pub first_mip: u32,
	pub mip_count: u32,
}

impl Default for Subresource {
	fn default() -> Self {
		Self {
			aspect: vk::ImageAspectFlags::COLOR,
			first_layer: 0,
			layer_count: vk::REMAINING_ARRAY_LAYERS,
			first_mip: 0,
			mip_count: vk::REMAINING_MIP_LEVELS,
		}
	}
}

/// A description for an image view.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewDesc<'a> {
	pub name: &'a str,
	pub image: vk::Image,
	pub view_type: vk::ImageViewType,
	pub format: vk::Format,
	pub usage: ImageViewUsage,
	pub size: vk::Extent3D,
	pub subresource: Subresource,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageViewDescUnnamed {
	pub image: vk::Image,
	pub view_type: vk::ImageViewType,
	pub format: vk::Format,
	pub usage: ImageViewUsage,
	pub size: vk::Extent3D,
	pub subresource: Subresource,
}

impl<'a> Named<'a> for ImageViewDesc<'a> {
	fn name(&self) -> &'a str { self.name }
}

impl ToNamed for ImageViewDescUnnamed {
	type Named<'a> = ImageViewDesc<'a>;

	fn to_named(self, name: &str) -> Self::Named<'_> {
		Self::Named {
			name,
			image: self.image,
			view_type: self.view_type,
			format: self.format,
			usage: self.usage,
			size: self.size,
			subresource: self.subresource,
		}
	}
}

/// A GPU-side image view.
#[derive(Default, Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageView {
	pub image: vk::Image,
	pub view: vk::ImageView,
	pub id: Option<ImageId>,
	pub storage_id: Option<StorageImageId>,
	pub size: vk::Extent3D,
}

impl Resource for ImageView {
	type Desc<'a> = ImageViewDesc<'a>;
	type Handle = Self;
	type UnnamedDesc = ImageViewDescUnnamed;

	fn handle(&self) -> Self::Handle { *self }

	fn create(device: &Device, desc: Self::Desc<'_>) -> Result<Self> {
		unsafe {
			let view = device.device().create_image_view(
				&vk::ImageViewCreateInfo::default()
					.image(desc.image)
					.view_type(desc.view_type)
					.format(desc.format)
					.components(vk::ComponentMapping {
						r: vk::ComponentSwizzle::IDENTITY,
						g: vk::ComponentSwizzle::IDENTITY,
						b: vk::ComponentSwizzle::IDENTITY,
						a: vk::ComponentSwizzle::IDENTITY,
					})
					.subresource_range(vk::ImageSubresourceRange {
						aspect_mask: desc.subresource.aspect,
						base_mip_level: desc.subresource.first_mip,
						level_count: desc.subresource.mip_count,
						base_array_layer: desc.subresource.first_layer,
						layer_count: desc.subresource.layer_count,
					}),
				None,
			)?;

			let _ = device.debug_utils_ext().map(|d| {
				let name = CString::new(desc.name).unwrap();
				d.set_debug_utils_object_name(
					&vk::DebugUtilsObjectNameInfoEXT::default()
						.object_handle(view)
						.object_name(&name),
				)
			});

			let (id, storage_id) = match desc.usage {
				ImageViewUsage::None => (None, None),
				ImageViewUsage::Sampled => (Some(device.descriptors().get_image(device, view)), None),
				ImageViewUsage::Storage => (None, Some(device.descriptors().get_storage_image(device, view))),
				ImageViewUsage::Both => (
					Some(device.descriptors().get_image(device, view)),
					Some(device.descriptors().get_storage_image(device, view)),
				),
			};

			Ok(Self {
				image: desc.image,
				view,
				id,
				storage_id,
				size: desc.size,
			})
		}
	}

	unsafe fn destroy(self, device: &Device) {
		unsafe {
			if let Some(id) = self.id {
				device.descriptors().return_image(id);
			}
			if let Some(id) = self.storage_id {
				device.descriptors().return_storage_image(id);
			}
			device.device().destroy_image_view(self.view, None);
		}
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ASDesc<'a> {
	pub name: &'a str,
	pub flags: vk::AccelerationStructureCreateFlagsKHR,
	pub ty: vk::AccelerationStructureTypeKHR,
	pub size: u64,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ASDescUnnamed {
	pub flags: vk::AccelerationStructureCreateFlagsKHR,
	pub ty: vk::AccelerationStructureTypeKHR,
	pub size: u64,
}

impl<'a> Named<'a> for ASDesc<'a> {
	fn name(&self) -> &'a str { self.name }
}

impl ToNamed for ASDescUnnamed {
	type Named<'a> = ASDesc<'a>;

	fn to_named(self, name: &str) -> Self::Named<'_> {
		Self::Named {
			name,
			flags: self.flags,
			ty: self.ty,
			size: self.size,
		}
	}
}

#[derive(Default)]
pub struct AS {
	pub inner: vk::AccelerationStructureKHR,
	pub buffer: Buffer,
}

impl Resource for AS {
	type Desc<'a> = ASDesc<'a>;
	type Handle = vk::AccelerationStructureKHR;
	type UnnamedDesc = ASDescUnnamed;

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, desc: Self::Desc<'_>) -> Result<Self> {
		unsafe {
			let buffer = Buffer::create(
				device,
				BufferDesc {
					name: desc.name,
					size: desc.size,
					readback: false,
				},
			)?;
			let inner = device.as_ext().create_acceleration_structure(
				&vk::AccelerationStructureCreateInfoKHR::default()
					.create_flags(desc.flags)
					.buffer(buffer.inner)
					.offset(0)
					.size(desc.size)
					.ty(desc.ty),
				None,
			)?;
			let _ = device.debug_utils_ext().map(|d| {
				let name = CString::new(desc.name).unwrap();
				d.set_debug_utils_object_name(
					&vk::DebugUtilsObjectNameInfoEXT::default()
						.object_handle(inner)
						.object_name(&name),
				)
			});

			Ok(Self { inner, buffer })
		}
	}

	unsafe fn destroy(self, device: &Device) {
		device.as_ext().destroy_acceleration_structure(self.inner, None);
		self.buffer.destroy(device);
	}
}

impl<'a> Named<'a> for () {
	fn name(&self) -> &'a str { "event" }
}

impl ToNamed for () {
	type Named<'a> = ();

	fn to_named(self, _: &str) -> Self::Named<'_> { () }
}

#[derive(Default)]
pub struct Event {
	inner: vk::Event,
}

impl Resource for Event {
	type Desc<'a> = ();
	type Handle = vk::Event;
	type UnnamedDesc = ();

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, _: Self::Desc<'_>) -> Result<Self> {
		unsafe {
			let inner = device.device().create_event(
				&vk::EventCreateInfo::default().flags(vk::EventCreateFlags::DEVICE_ONLY),
				None,
			)?;
			Ok(Self { inner })
		}
	}

	unsafe fn destroy(self, device: &Device) { device.device().destroy_event(self.inner, None); }
}
