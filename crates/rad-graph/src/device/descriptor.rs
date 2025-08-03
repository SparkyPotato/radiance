//! Abstractions around descriptor indexing, for efficient and easy resource access on the GPU.

use std::{collections::VecDeque, num::NonZeroU32, sync::Mutex};

use ash::vk;

use crate::Result;

/// An ID representing a sampled image, for use by a shader.
///
/// Is a `u32`, bound to binding `0`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageId(NonZeroU32);
impl ImageId {
	pub fn get(&self) -> u32 { self.0.get() }

	pub unsafe fn from_raw(raw: u32) -> Self { unsafe { Self(NonZeroU32::new_unchecked(raw)) }}
}
/// An ID representing a storage image, for use by a shader.
///
/// Is a `u32`, bound to binding `1`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageImageId(NonZeroU32);
/// An ID representing a sampler, for use by a shader.
///
/// Is a `u32`, bound to binding `2`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(NonZeroU32);

#[cfg(feature = "bytemuck")]
mod bytemuck {
	use super::*;

	unsafe impl ::bytemuck::NoUninit for ImageId {}
	unsafe impl ::bytemuck::PodInOption for ImageId {}
	unsafe impl ::bytemuck::ZeroableInOption for ImageId {}

	unsafe impl ::bytemuck::NoUninit for StorageImageId {}
	unsafe impl ::bytemuck::PodInOption for StorageImageId {}
	unsafe impl ::bytemuck::ZeroableInOption for StorageImageId {}

	unsafe impl ::bytemuck::NoUninit for SamplerId {}
	unsafe impl ::bytemuck::PodInOption for SamplerId {}
	unsafe impl ::bytemuck::ZeroableInOption for SamplerId {}
}

pub struct Descriptors {
	pool: vk::DescriptorPool,
	layout: vk::PipelineLayout,
	set: vk::DescriptorSet,
	set_layout: vk::DescriptorSetLayout,
	inner: Mutex<Inner>,
}

struct Inner {
	sampled_images: FreeIndices,
	storage_images: FreeIndices,
	samplers: FreeIndices,
}

impl Descriptors {
	pub fn set(&self) -> vk::DescriptorSet { self.set }

	/// Get a `PipelineLayout` that should be used when making pipelines.
	pub fn layout(&self) -> vk::PipelineLayout { self.layout }

	pub fn get_image(&self, device: &ash::Device, image: vk::ImageView) -> ImageId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.sampled_images.get_index();
		unsafe {
			device.update_descriptor_sets(
				&[vk::WriteDescriptorSet::default()
					.dst_set(self.set)
					.dst_binding(0)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
					.image_info(&[vk::DescriptorImageInfo::default()
						.image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
						.image_view(image)])],
				&[],
			);
		}

		ImageId(index)
	}

	pub fn get_storage_image(&self, device: &ash::Device, image: vk::ImageView) -> StorageImageId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.storage_images.get_index();
		unsafe {
			device.update_descriptor_sets(
				&[vk::WriteDescriptorSet::default()
					.dst_set(self.set)
					.dst_binding(1)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
					.image_info(&[vk::DescriptorImageInfo::default()
						.image_layout(vk::ImageLayout::GENERAL)
						.image_view(image)])],
				&[],
			);
		}

		StorageImageId(index)
	}

	pub fn get_sampler(&self, device: &ash::Device, sampler: vk::Sampler) -> SamplerId {
		let mut inner = self.inner.lock().unwrap();

		let index = inner.samplers.get_index();
		unsafe {
			device.update_descriptor_sets(
				&[vk::WriteDescriptorSet::default()
					.dst_set(self.set)
					.dst_binding(2)
					.dst_array_element(index.get())
					.descriptor_type(vk::DescriptorType::SAMPLER)
					.image_info(&[vk::DescriptorImageInfo::default().sampler(sampler)])],
				&[],
			);
		}

		SamplerId(index)
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

	pub(super) fn new(device: &ash::Device) -> Result<Self> {
		let sampled_image_count = 512 * 1024;
		let storage_image_count = 512 * 1024;
		let sampler_count = 512;

		let binding_flags = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
			| vk::DescriptorBindingFlags::PARTIALLY_BOUND
			| vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;

		let set_layout = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
				.descriptor_count(sampled_image_count)
				.stage_flags(vk::ShaderStageFlags::ALL),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(storage_image_count)
				.stage_flags(vk::ShaderStageFlags::ALL),
			vk::DescriptorSetLayoutBinding::default()
				.binding(2)
				.descriptor_type(vk::DescriptorType::SAMPLER)
				.descriptor_count(sampler_count)
				.stage_flags(vk::ShaderStageFlags::ALL),
		];

		unsafe {
			let set_layout = device.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo::default()
					.bindings(&set_layout)
					.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
					.push_next(
						&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
							binding_flags,
							binding_flags,
							binding_flags,
						]),
					),
				None,
			)?;

			let pool = device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::default()
					.max_sets(1)
					.pool_sizes(&[
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::SAMPLED_IMAGE)
							.descriptor_count(sampled_image_count),
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(storage_image_count),
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::SAMPLER)
							.descriptor_count(sampler_count),
					])
					.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
				None,
			)?;

			let set = device.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::default()
					.descriptor_pool(pool)
					.set_layouts(&[set_layout]),
			)?[0];

			let layout = device.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[set_layout])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.size(128)
						.offset(0)
						.stage_flags(vk::ShaderStageFlags::ALL)]),
				None,
			)?;

			Ok(Descriptors {
				pool,
				layout,
				set,
				set_layout,
				inner: Mutex::new(Inner {
					sampled_images: FreeIndices::new(sampled_image_count),
					storage_images: FreeIndices::new(storage_image_count),
					samplers: FreeIndices::new(sampler_count),
				}),
			})
		}
	}

	pub(super) unsafe fn cleanup(&mut self, device: &ash::Device) { unsafe {
		device.destroy_pipeline_layout(self.layout, None);
		device.destroy_descriptor_set_layout(self.set_layout, None);
		device.destroy_descriptor_pool(self.pool, None);
	}}
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
