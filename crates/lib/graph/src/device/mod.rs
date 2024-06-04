//! An abstraction over a raw Vulkan device.

use std::{
	mem::ManuallyDrop,
	sync::{Mutex, MutexGuard},
};

use ash::{
	extensions::{ext, khr},
	vk,
};
pub use gpu_allocator::vulkan as alloc;
use gpu_allocator::vulkan::Allocator;

pub use crate::device::queue::{ComputeSyncPoint, GraphicsSyncPoint, QueueType, Queues, SyncPoint, TransferSyncPoint};
use crate::{
	device::{
		descriptor::Descriptors,
		queue::{Queue, SubmitBuilder},
	},
	Result,
};

pub mod descriptor;
mod init;
mod queue;

/// Has everything you need to do Vulkan stuff.
pub struct Device {
	debug_messenger: vk::DebugUtilsMessengerEXT, // Can be null.
	physical_device: vk::PhysicalDevice,
	device: ash::Device,
	as_ext: khr::AccelerationStructure,
	rt_ext: khr::RayTracingPipeline,
	surface_ext: Option<khr::Surface>,
	debug_utils_ext: Option<ext::DebugUtils>,
	queues: Queues<Queue>,
	allocator: ManuallyDrop<Mutex<Allocator>>,
	descriptors: Descriptors,
	instance: ash::Instance,
	entry: ash::Entry,
}

impl Device {
	pub fn entry(&self) -> &ash::Entry { &self.entry }

	pub fn instance(&self) -> &ash::Instance { &self.instance }

	pub fn device(&self) -> &ash::Device { &self.device }

	pub fn physical_device(&self) -> vk::PhysicalDevice { self.physical_device }

	pub fn as_ext(&self) -> &khr::AccelerationStructure { &self.as_ext }

	pub fn rt_ext(&self) -> &khr::RayTracingPipeline { &self.rt_ext }

	pub fn surface_ext(&self) -> Option<&khr::Surface> { self.surface_ext.as_ref() }

	pub fn debug_utils_ext(&self) -> Option<&ext::DebugUtils> { self.debug_utils_ext.as_ref() }

	pub fn queue_families(&self) -> Queues<u32> { self.queues.map_ref(|data| data.family) }

	pub fn allocator(&self) -> MutexGuard<'_, Allocator> { self.allocator.lock().unwrap() }

	pub fn descriptors(&self) -> &Descriptors { &self.descriptors }

	pub unsafe fn submit<'a, 'q, const TY: QueueType>(
		&'q self, submits: &'a [vk::SubmitInfo2],
	) -> SubmitBuilder<'a, 'q, TY> {
		let q = self.queues.get(TY);
		SubmitBuilder::new(&self.device, submits, q)
	}

	pub unsafe fn submit_graphics<'a, 'q>(
		&'q self, submits: &'a [vk::SubmitInfo2],
	) -> SubmitBuilder<'a, 'q, { QueueType::Graphics }> {
		self.submit(submits)
	}

	pub unsafe fn submit_compute<'a, 'q>(
		&'q self, submits: &'a [vk::SubmitInfo2],
	) -> SubmitBuilder<'a, 'q, { QueueType::Compute }> {
		self.submit(submits)
	}

	pub unsafe fn submit_transfer<'a, 'q>(
		&'q self, submits: &'a [vk::SubmitInfo2],
	) -> SubmitBuilder<'a, 'q, { QueueType::Transfer }> {
		self.submit(submits)
	}

	pub unsafe fn wait<const TY: QueueType>(&self, point: SyncPoint<TY>) -> Result<()> {
		let q = self.queues.get(TY);
		self.device().wait_semaphores(
			&vk::SemaphoreWaitInfo::builder()
				.semaphores(&[q.semaphore])
				.values(&[point.value])
				.build(),
			u64::MAX,
		)?;
		Ok(())
	}

	pub unsafe fn is_synced<const TY: QueueType>(&self, point: SyncPoint<TY>) -> Result<bool> {
		let q = self.queues.get(TY);
		let val = self.device().get_semaphore_counter_value(q.semaphore)?;
		Ok(val >= point.value)
	}
}

impl Drop for Device {
	fn drop(&mut self) {
		unsafe {
			// Drop the allocator before the device.
			ManuallyDrop::drop(&mut self.allocator);
			self.queues.map_ref(|q| q.destroy(self.device()));
			self.descriptors.cleanup(&self.device);

			self.device.destroy_device(None);

			if let Some(utils) = self.debug_utils_ext.as_ref() {
				utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}
