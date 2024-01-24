//! Abstractions around descriptor indexing, for efficient and easy resource access on the GPU.

use std::{collections::VecDeque, num::NonZeroU32, sync::Mutex};

use ash::vk;

use crate::{device::Device, Result};

/// An ID representing a storage buffer, for use by a shader.
///
/// Is a `u32`, bound to binding `0`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(NonZeroU32);
/// An ID representing a sampled image, for use by a shader.
///
/// Is a `u32`, bound to binding `1`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageId(NonZeroU32);
/// An ID representing a storage image, for use by a shader.
///
/// Is a `u32`, bound to binding `2`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageImageId(NonZeroU32);
/// An ID representing a sampler, for use by a shader.
///
/// Is a `u32`, bound to binding `3`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(NonZeroU32);
/// An ID representing an acceleration structure, for use by a shader.
///
/// Is a `u32`, bound to binding `4`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ASId(NonZeroU32);

#[cfg(feature = "bytemuck")]
mod bytemuck {
	use super::*;

	unsafe impl ::bytemuck::NoUninit for BufferId {}
	unsafe impl ::bytemuck::PodInOption for BufferId {}
	unsafe impl ::bytemuck::ZeroableInOption for BufferId {}

	unsafe impl ::bytemuck::NoUninit for ImageId {}
	unsafe impl ::bytemuck::PodInOption for ImageId {}
	unsafe impl ::bytemuck::ZeroableInOption for ImageId {}

	unsafe impl ::bytemuck::NoUninit for StorageImageId {}
	unsafe impl ::bytemuck::PodInOption for StorageImageId {}
	unsafe impl ::bytemuck::ZeroableInOption for StorageImageId {}

	unsafe impl ::bytemuck::NoUninit for SamplerId {}
	unsafe impl ::bytemuck::PodInOption for SamplerId {}
	unsafe impl ::bytemuck::ZeroableInOption for SamplerId {}

	unsafe impl ::bytemuck::NoUninit for ASId {}
	unsafe impl ::bytemuck::PodInOption for ASId {}
	unsafe impl ::bytemuck::ZeroableInOption for ASId {}
}

pub struct Descriptors {
	pool: vk::DescriptorPool,
	layout: vk::DescriptorSetLayout,
	set: vk::DescriptorSet,
	inner: Mutex<Inner>,
}

struct Inner {
	storage_buffers: FreeIndices,
	sampled_images: FreeIndices,
	storage_images: FreeIndices,
	samplers: FreeIndices,
	ases: FreeIndices,
}

impl Descriptors {
	pub fn set(&self) -> vk::DescriptorSet { self.set }

	/// Get a `DescriptorSetLayout` that should be used when making pipelines.
	pub fn layout(&self) -> vk::DescriptorSetLayout { self.layout }

	pub fn get_buffer(&self, device: &Device, buffer: vk::Buffer) -> BufferId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.storage_buffers.get_index();
		unsafe {
			device.device().update_descriptor_sets(
				&[vk::WriteDescriptorSet::builder()
					.dst_set(self.set)
					.dst_binding(0)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
					.buffer_info(&[vk::DescriptorBufferInfo::builder()
						.buffer(buffer)
						.offset(0)
						.range(vk::WHOLE_SIZE)
						.build()])
					.build()],
				&[],
			);
		}

		BufferId(index)
	}

	pub fn get_image(&self, device: &Device, image: vk::ImageView) -> ImageId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.sampled_images.get_index();
		unsafe {
			device.device().update_descriptor_sets(
				&[vk::WriteDescriptorSet::builder()
					.dst_set(self.set)
					.dst_binding(1)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
					.image_info(&[vk::DescriptorImageInfo::builder()
						.image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
						.image_view(image)
						.build()])
					.build()],
				&[],
			);
		}

		ImageId(index)
	}

	pub fn get_storage_image(&self, device: &Device, image: vk::ImageView) -> StorageImageId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.storage_images.get_index();
		unsafe {
			device.device().update_descriptor_sets(
				&[vk::WriteDescriptorSet::builder()
					.dst_set(self.set)
					.dst_binding(2)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
					.image_info(&[vk::DescriptorImageInfo::builder()
						.image_layout(vk::ImageLayout::GENERAL)
						.image_view(image)
						.build()])
					.build()],
				&[],
			);
		}

		StorageImageId(index)
	}

	pub fn get_sampler(&self, device: &Device, sampler: vk::Sampler) -> SamplerId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.samplers.get_index();
		unsafe {
			device.device().update_descriptor_sets(
				&[vk::WriteDescriptorSet::builder()
					.dst_set(self.set)
					.dst_binding(3)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::SAMPLER)
					.image_info(&[vk::DescriptorImageInfo::builder().sampler(sampler).build()])
					.build()],
				&[],
			);
		}

		SamplerId(index)
	}

	pub fn get_as(&self, device: &Device, as_: vk::AccelerationStructureKHR) -> ASId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.ases.get_index();
		unsafe {
			let mut ds = vk::WriteDescriptorSet::builder();
			ds.descriptor_count = 1;
			device.device().update_descriptor_sets(
				&[ds.dst_set(self.set)
					.dst_binding(4)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
					.push_next(
						&mut vk::WriteDescriptorSetAccelerationStructureKHR::builder().acceleration_structures(&[as_]),
					)
					.build()],
				&[],
			);
		}

		ASId(index)
	}

	pub fn return_buffer(&self, index: BufferId) {
		let mut inner = self.inner.lock().unwrap();
		inner.storage_buffers.return_index(index.0);
	}

	pub fn return_image(&self, index: ImageId) {
		let mut inner = self.inner.lock().unwrap();
		inner.sampled_images.return_index(index.0);
	}

	pub fn return_storage_image(&self, index: StorageImageId) {
		let mut inner = self.inner.lock().unwrap();
		inner.storage_images.return_index(index.0);
	}

	pub fn return_sampler(&self, index: SamplerId) {
		let mut inner = self.inner.lock().unwrap();
		inner.samplers.return_index(index.0);
	}

	pub fn return_as(&self, index: ASId) {
		let mut inner = self.inner.lock().unwrap();
		inner.ases.return_index(index.0);
	}

	pub(super) fn new(device: &ash::Device) -> Result<Self> {
		let storage_buffer_count = 512 * 1024;
		let sampled_image_count = 512 * 1024;
		let storage_image_count = 512 * 1024;
		let sampler_count = 512;
		let as_count = 512 * 1024;

		let binding_flags = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
			| vk::DescriptorBindingFlags::PARTIALLY_BOUND
			| vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;

		let set_layout = [
			vk::DescriptorSetLayoutBinding::builder()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(storage_buffer_count)
				.stage_flags(vk::ShaderStageFlags::ALL)
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(1)
				.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
				.descriptor_count(sampled_image_count)
				.stage_flags(vk::ShaderStageFlags::ALL)
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(2)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(storage_image_count)
				.stage_flags(vk::ShaderStageFlags::ALL)
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(3)
				.descriptor_type(vk::DescriptorType::SAMPLER)
				.descriptor_count(sampler_count)
				.stage_flags(vk::ShaderStageFlags::ALL)
				.build(),
			vk::DescriptorSetLayoutBinding::builder()
				.binding(4)
				.descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
				.descriptor_count(as_count)
				.stage_flags(vk::ShaderStageFlags::ALL)
				.build(),
		];
		let layout = unsafe {
			device.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo::builder()
					.bindings(&set_layout)
					.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
					.push_next(
						&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&[
							binding_flags,
							binding_flags,
							binding_flags,
							binding_flags,
							binding_flags,
						]),
					),
				None,
			)?
		};

		let pool = unsafe {
			device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::builder()
					.max_sets(1)
					.pool_sizes(&[
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_BUFFER)
							.descriptor_count(storage_buffer_count)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::SAMPLED_IMAGE)
							.descriptor_count(sampled_image_count)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(storage_image_count)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::SAMPLER)
							.descriptor_count(sampler_count)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
							.descriptor_count(as_count)
							.build(),
					])
					.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
				None,
			)?
		};

		let set = unsafe {
			device.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::builder()
					.descriptor_pool(pool)
					.set_layouts(&[layout]),
			)?[0]
		};

		Ok(Descriptors {
			pool,
			layout,
			set,
			inner: Mutex::new(Inner {
				storage_buffers: FreeIndices::new(storage_buffer_count),
				sampled_images: FreeIndices::new(sampled_image_count),
				storage_images: FreeIndices::new(storage_image_count),
				samplers: FreeIndices::new(sampler_count),
				ases: FreeIndices::new(as_count),
			}),
		})
	}

	pub(super) unsafe fn cleanup(&mut self, device: &ash::Device) {
		device.destroy_descriptor_set_layout(self.layout, None);
		device.destroy_descriptor_pool(self.pool, None);
	}
}

struct FreeIndices {
	max: u32,
	unallocated: u32,
	returned: VecDeque<NonZeroU32>,
}

impl FreeIndices {
	pub fn new(max: u32) -> Self {
		Self {
			max,
			unallocated: 1,
			returned: VecDeque::new(),
		}
	}

	pub fn get_index(&mut self) -> NonZeroU32 {
		if let Some(index) = self.returned.pop_back() {
			index
		} else {
			let v = self.unallocated;
			self.unallocated += 1;
			assert!(v < self.max, "too many descriptor indices allocated");
			unsafe { NonZeroU32::new_unchecked(v) }
		}
	}

	pub fn return_index(&mut self, index: NonZeroU32) { self.returned.push_front(index) }
}
