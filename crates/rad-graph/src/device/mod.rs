//! An abstraction over a raw Vulkan device.

use std::{
	mem::ManuallyDrop,
	sync::{Arc, Mutex, MutexGuard},
};

use ash::{ext, khr, vk};
pub use gpu_allocator::vulkan as alloc;
use gpu_allocator::vulkan::Allocator;

pub use crate::device::{
	queue::{
		Compute,
		Graphics,
		QueueSyncs,
		QueueType,
		QueueWait,
		QueueWaitOwned,
		Queues,
		SyncPoint,
		SyncStage,
		Transfer,
	},
	sampler::SamplerDesc,
	shader::{ComputePipeline, GraphicsPipeline, GraphicsPipelineDesc, HotreloadStatus, ShaderInfo},
};
use crate::{
	device::{
		descriptor::{Descriptors, SamplerId},
		queue::QueueData,
		sampler::Samplers,
		shader::ShaderRuntime,
	},
	Result,
};

pub mod descriptor;
mod init;
mod queue;
mod sampler;
mod shader;

struct DeviceInner {
	physical_device: vk::PhysicalDevice,
	device: ash::Device,
	as_ext: khr::acceleration_structure::Device,
	rt_ext: khr::ray_tracing_pipeline::Device,
	surface_ext: khr::surface::Instance,
	debug_utils_ext: Option<ext::debug_utils::Device>,
	queues: Queues<QueueData>,
	allocator: ManuallyDrop<Mutex<Allocator>>,
	shaders: ManuallyDrop<ShaderRuntime>,
	descriptors: Descriptors,
	samplers: Mutex<Samplers>,
	instance: ash::Instance,
	entry: ash::Entry,
}

/// Has everything you need to do Vulkan stuff.
#[derive(Clone)]
pub struct Device {
	inner: Arc<DeviceInner>,
}

impl Device {
	#[track_caller]
	pub fn graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> Result<GraphicsPipeline> {
		self.inner.shaders.create_graphics_pipeline(desc).map_err(Into::into)
	}

	#[track_caller]
	pub fn compute_pipeline(&self, shader: ShaderInfo) -> Result<ComputePipeline> {
		self.inner.shaders.create_compute_pipeline(shader).map_err(Into::into)
	}

	pub fn get_shader(&self, info: ShaderInfo) -> std::result::Result<(Vec<u32>, vk::ShaderStageFlags), String> {
		self.inner.shaders.get_shader(info)
	}

	pub fn layout(&self) -> vk::PipelineLayout { self.inner.descriptors.layout() }

	pub fn hotreload_status(&self) -> HotreloadStatus { self.inner.shaders.status() }

	pub fn entry(&self) -> &ash::Entry { &self.inner.entry }

	pub fn instance(&self) -> &ash::Instance { &self.inner.instance }

	pub fn device(&self) -> &ash::Device { &self.inner.device }

	pub fn physical_device(&self) -> vk::PhysicalDevice { self.inner.physical_device }

	pub fn as_ext(&self) -> &khr::acceleration_structure::Device { &self.inner.as_ext }

	pub fn rt_ext(&self) -> &khr::ray_tracing_pipeline::Device { &self.inner.rt_ext }

	pub fn surface_ext(&self) -> &khr::surface::Instance { &self.inner.surface_ext }

	pub fn debug_utils_ext(&self) -> Option<&ext::debug_utils::Device> { self.inner.debug_utils_ext.as_ref() }

	pub fn allocator(&self) -> MutexGuard<'_, Allocator> { self.inner.allocator.lock().unwrap() }

	pub fn descriptor_set(&self) -> vk::DescriptorSet { self.inner.descriptors.set() }

	pub fn image_id(&self, image: vk::ImageView) -> descriptor::ImageId {
		self.inner.descriptors.get_image(&self.inner.device, image)
	}

	pub fn return_image_id(&self, id: descriptor::ImageId) { self.inner.descriptors.return_image(id) }

	pub fn storage_image_id(&self, image: vk::ImageView) -> descriptor::StorageImageId {
		self.inner.descriptors.get_storage_image(&self.inner.device, image)
	}

	pub fn return_storage_image_id(&self, id: descriptor::StorageImageId) {
		self.inner.descriptors.return_storage_image(id)
	}

	pub fn sampler(&self, desc: SamplerDesc) -> SamplerId {
		self.inner
			.samplers
			.lock()
			.unwrap()
			.get(&self.inner.device, &self.inner.descriptors, desc)
	}

	pub fn queue_families(&self) -> Queues<u32> { self.inner.queues.map_ref(|data| data.family()) }

	pub fn queue<TY: QueueType>(&self) -> MutexGuard<'_, vk::Queue> { self.inner.queues.get::<TY>().queue() }

	pub fn current_sync_point<TY: QueueType>(&self) -> SyncPoint<TY> { self.inner.queues.get::<TY>().current() }

	pub fn submit<TY: QueueType>(
		&self, wait: QueueWait, bufs: &[vk::CommandBuffer], signal: &[SyncStage<vk::Semaphore>], fence: vk::Fence,
	) -> Result<SyncPoint<TY>> {
		self.inner
			.queues
			.get::<TY>()
			.submit(&self.inner.queues, self, wait, bufs, signal, fence)
	}
}

impl Drop for DeviceInner {
	fn drop(&mut self) {
		unsafe {
			// Drop the allocator before the device.
			ManuallyDrop::drop(&mut self.allocator);
			ManuallyDrop::drop(&mut self.shaders);
			self.samplers.get_mut().unwrap().cleanup(&self.device);
			self.descriptors.cleanup(&self.device);
			self.queues.map_ref(|x| x.destroy(&self.device));

			self.device.destroy_device(None);

			self.instance.destroy_instance(None);
		}
	}
}
