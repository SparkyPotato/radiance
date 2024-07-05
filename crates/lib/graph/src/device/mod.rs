//! An abstraction over a raw Vulkan device.

use std::{
	ffi::CStr,
	mem::ManuallyDrop,
	sync::{Mutex, MutexGuard},
};

use ash::{
	extensions::{ext, khr},
	vk,
};
pub use gpu_allocator::vulkan as alloc;
use gpu_allocator::vulkan::Allocator;
use radiance_shader_compiler::runtime::ShaderRuntime;

pub use crate::device::queue::{
	Compute,
	Graphics,
	QueueType,
	QueueWait,
	QueueWaitOwned,
	Queues,
	SyncPoint,
	SyncStage,
	Transfer,
};
use crate::{
	arena::Arena,
	device::{descriptor::Descriptors, queue::QueueData},
	Result,
};

pub mod descriptor;
mod init;
mod queue;

/// Has everything you need to do Vulkan stuff.
pub struct Device {
	arena: Arena,
	debug_messenger: vk::DebugUtilsMessengerEXT, // Can be null.
	physical_device: vk::PhysicalDevice,
	device: ash::Device,
	as_ext: khr::AccelerationStructure,
	rt_ext: khr::RayTracingPipeline,
	surface_ext: Option<khr::Surface>,
	debug_utils_ext: Option<ext::DebugUtils>,
	queues: Queues<QueueData>,
	allocator: ManuallyDrop<Mutex<Allocator>>,
	shaders: ManuallyDrop<ShaderRuntime>,
	descriptors: Descriptors,
	instance: ash::Instance,
	entry: ash::Entry,
}

impl Device {
	pub fn arena(&self) -> &Arena { &self.arena }

	pub fn reset_arena(&mut self) { self.arena.reset() }

	pub fn shader<'a>(
		&'a self, name: &'a CStr, stage: vk::ShaderStageFlags, specialization: Option<&'a vk::SpecializationInfo>,
	) -> vk::PipelineShaderStageCreateInfoBuilder {
		self.shaders.shader(name, stage, specialization)
	}

	pub fn entry(&self) -> &ash::Entry { &self.entry }

	pub fn instance(&self) -> &ash::Instance { &self.instance }

	pub fn device(&self) -> &ash::Device { &self.device }

	pub fn physical_device(&self) -> vk::PhysicalDevice { self.physical_device }

	pub fn as_ext(&self) -> &khr::AccelerationStructure { &self.as_ext }

	pub fn rt_ext(&self) -> &khr::RayTracingPipeline { &self.rt_ext }

	pub fn surface_ext(&self) -> Option<&khr::Surface> { self.surface_ext.as_ref() }

	pub fn debug_utils_ext(&self) -> Option<&ext::DebugUtils> { self.debug_utils_ext.as_ref() }

	pub fn allocator(&self) -> MutexGuard<'_, Allocator> { self.allocator.lock().unwrap() }

	pub fn descriptors(&self) -> &Descriptors { &self.descriptors }

	pub fn queue_families(&self) -> Queues<u32> { self.queues.map_ref(|data| data.family()) }

	pub fn queue<TY: QueueType>(&self) -> MutexGuard<'_, vk::Queue> { self.queues.get::<TY>().queue() }

	pub fn current_sync_point<TY: QueueType>(&self) -> SyncPoint<TY> { self.queues.get::<TY>().current() }

	pub fn submit<TY: QueueType>(
		&self, wait: QueueWait, bufs: &[vk::CommandBuffer], signal: &[SyncStage<vk::Semaphore>], fence: vk::Fence,
	) -> Result<SyncPoint<TY>> {
		self.queues
			.get::<TY>()
			.submit(&self.queues, self, wait, bufs, signal, fence)
	}
}

impl Drop for Device {
	fn drop(&mut self) {
		unsafe {
			// Drop the allocator before the device.
			ManuallyDrop::drop(&mut self.allocator);
			ManuallyDrop::take(&mut self.shaders).destroy(&self.device);
			self.descriptors.cleanup(&self.device);
			self.queues.map_ref(|x| x.destroy(&self.device));

			self.device.destroy_device(None);

			if let Some(utils) = self.debug_utils_ext.as_ref() {
				utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}
