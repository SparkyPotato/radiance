//! An abstraction over a raw Vulkan device.

use std::{
	mem::ManuallyDrop,
	sync::{Mutex, MutexGuard},
};

use ash::{
	extensions::{ext::DebugUtils, khr::Surface},
	vk::{DebugUtilsMessengerEXT, Fence, PhysicalDevice, Queue, SubmitInfo2},
	Entry,
	Instance,
};
use gpu_allocator::vulkan::Allocator;

use crate::{device::descriptor::Descriptors, Result};

pub mod cmd;
pub mod descriptor;
mod init;

/// Has everything you need to do Vulkan stuff.
pub struct Device {
	debug_messenger: DebugUtilsMessengerEXT, // Can be null.
	physical_device: PhysicalDevice,
	device: ash::Device,
	surface_ext: Option<Surface>,
	debug_utils_ext: Option<DebugUtils>,
	queues: Queues<QueueData>,
	allocator: ManuallyDrop<Mutex<Allocator>>,
	descriptors: Descriptors,
	instance: Instance,
	entry: Entry,
}

struct QueueData {
	queue: Mutex<Queue>,
	family: u32,
}

/// Data consisting of two queue strategies:
/// - Separate: Separate queues for graphics and presentation, async compute, and DMA transfer.
/// - Single: One queue for all operations.
pub enum Queues<T> {
	Separate {
		graphics: T, // Also supports presentation.
		compute: T,
		transfer: T,
	},
	Single(T),
}

impl<T> Queues<T> {
	fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> Queues<U> {
		match self {
			Queues::Separate {
				graphics,
				compute,
				transfer,
			} => Queues::Separate {
				graphics: f(graphics),
				compute: f(compute),
				transfer: f(transfer),
			},
			Queues::Single(queue) => Queues::Single(f(queue)),
		}
	}

	pub fn graphics(&self) -> &T {
		match self {
			Queues::Separate { graphics, .. } => graphics,
			Queues::Single(queue) => queue,
		}
	}

	pub fn compute(&self) -> &T {
		match self {
			Queues::Separate { compute, .. } => compute,
			Queues::Single(queue) => queue,
		}
	}

	pub fn transfer(&self) -> &T {
		match self {
			Queues::Separate { transfer, .. } => transfer,
			Queues::Single(queue) => queue,
		}
	}
}

impl Device {
	pub fn entry(&self) -> &Entry { &self.entry }

	pub fn instance(&self) -> &Instance { &self.instance }

	pub fn device(&self) -> &ash::Device { &self.device }

	pub fn physical_device(&self) -> PhysicalDevice { self.physical_device }

	pub fn surface_ext(&self) -> Option<&Surface> { self.surface_ext.as_ref() }

	pub fn queue_families(&self) -> Queues<u32> { self.queues.map(|data| data.family) }

	pub fn graphics_queue(&self) -> MutexGuard<'_, Queue> { self.queues.graphics().queue.lock().unwrap() }

	pub fn compute_queue(&self) -> MutexGuard<'_, Queue> { self.queues.compute().queue.lock().unwrap() }

	pub fn transfer_queue(&self) -> MutexGuard<'_, Queue> { self.queues.transfer().queue.lock().unwrap() }

	pub fn allocator(&self) -> MutexGuard<'_, Allocator> { self.allocator.lock().unwrap() }

	pub fn base_descriptors(&self) -> &Descriptors { &self.descriptors }

	pub fn needs_queue_ownership_transfer(&self) -> bool {
		match self.queues {
			Queues::Separate { .. } => true,
			Queues::Single(_) => false,
		}
	}

	/// # Safety
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_graphics(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(graphics) => unsafe {
				self.device
					.queue_submit2(*graphics.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { graphics, .. } => unsafe {
				self.device
					.queue_submit2(*graphics.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}

	/// # Safety
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_compute(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(compute) => unsafe {
				self.device
					.queue_submit2(*compute.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { compute, .. } => unsafe {
				self.device
					.queue_submit2(*compute.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}

	/// # Safety
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_transfer(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(transfer) => unsafe {
				self.device
					.queue_submit2(*transfer.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { transfer, .. } => unsafe {
				self.device
					.queue_submit2(*transfer.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}
}

impl Drop for Device {
	fn drop(&mut self) {
		unsafe {
			// Drop the allocator before the device.
			ManuallyDrop::drop(&mut self.allocator);
			self.descriptors.cleanup(&self.device);

			self.device.destroy_device(None);

			if let Some(utils) = self.debug_utils_ext.as_ref() {
				utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}
