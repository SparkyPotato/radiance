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
	shader::{ComputePipeline, GraphicsPipeline, GraphicsPipelineDesc, HotreloadStatus, ShaderInfo},
};
use crate::{
	device::{descriptor::Descriptors, queue::QueueData, shader::ShaderRuntime},
	Result,
};

pub mod descriptor;
mod init;
mod queue;
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

	pub fn descriptors(&self) -> &Descriptors { &self.inner.descriptors }

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
			self.descriptors.cleanup(&self.device);
			self.queues.map_ref(|x| x.destroy(&self.device));

			self.device.destroy_device(None);

			self.instance.destroy_instance(None);
		}
	}
}
