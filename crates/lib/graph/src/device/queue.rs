use std::{marker::ConstParamTy, sync::Mutex};

use ash::vk;

use crate::{
	arena::{Arena, IteratorAlloc},
	Result,
};

#[derive(ConstParamTy, PartialEq, Eq, Copy, Clone)]
pub enum QueueType {
	Graphics,
	Compute,
	Transfer,
}

#[derive(Copy, Clone)]
pub struct SyncPoint<const TY: QueueType> {
	pub(super) value: u64,
}

impl<const TY: QueueType> SyncPoint<TY> {
	pub fn none() -> Self { Self { value: 0 } }
}

pub type GraphicsSyncPoint = SyncPoint<{ QueueType::Graphics }>;
pub type ComputeSyncPoint = SyncPoint<{ QueueType::Compute }>;
pub type TransferSyncPoint = SyncPoint<{ QueueType::Transfer }>;

pub struct SubmitBuilder<'a, 'q, const TY: QueueType> {
	device: &'q ash::Device,
	info: &'a [vk::SubmitInfo2],
	queue: &'q Queue,
	wait: vk::PipelineStageFlags2,
	sync: u64,
}

impl<'a, 'q, const TY: QueueType> SubmitBuilder<'a, 'q, TY> {
	pub(super) fn new(device: &'q ash::Device, info: &'a [vk::SubmitInfo2], queue: &'q Queue) -> Self {
		Self {
			device,
			info,
			queue,
			sync: 0,
			wait: vk::PipelineStageFlags2::empty(),
		}
	}

	pub fn wait_on_prev(mut self, stage: vk::PipelineStageFlags2) -> Self {
		self.wait = stage;
		self
	}

	pub fn wait_on(mut self, point: SyncPoint<TY>, stage: vk::PipelineStageFlags2) -> Self {
		self.wait = stage;
		self.sync = point.value;
		self
	}

	pub unsafe fn submit_signal(self, signal: vk::PipelineStageFlags2) -> Result<SyncPoint<TY>> {
		let sig = {
			let (_, v, _) = &mut *self.queue.queue.lock().unwrap();
			let wait_v = *v;
			let signal_v = wait_v + 1;
			*v = signal_v;
			signal_v
		};
		let signal = Some(
			vk::SemaphoreSubmitInfo::builder()
				.semaphore(self.queue.semaphore)
				.stage_mask(signal)
				.value(sig)
				.build(),
		);
		self.submit_inner(signal).map(|x| x.unwrap())
	}

	pub unsafe fn submit(self) -> Result<()> { self.submit_inner(None).map(|_| ()) }

	unsafe fn submit_inner(self, signal: Option<vk::SemaphoreSubmitInfo>) -> Result<Option<SyncPoint<TY>>> {
		let (queue, v, a) = &mut *self.queue.queue.lock().unwrap();

		{
			let a = &*a;
			let wait = (!self.wait.is_empty()).then_some(
				vk::SemaphoreSubmitInfo::builder()
					.semaphore(self.queue.semaphore)
					.stage_mask(self.wait)
					.value(*v)
					.build(),
			);

			let mut submits: Vec<_, _> = self.info.into_iter().copied().collect_in(a);
			let first = submits.first_mut().unwrap();
			let mut waits: Vec<_, _> =
				std::slice::from_raw_parts(first.p_wait_semaphore_infos, first.wait_semaphore_info_count as _)
					.into_iter()
					.copied()
					.collect_in(a);
			waits.extend(wait);
			first.p_wait_semaphore_infos = waits.as_ptr();
			first.wait_semaphore_info_count += 1;

			let last = submits.last_mut().unwrap();
			let mut signals: Vec<_, _> =
				std::slice::from_raw_parts(last.p_signal_semaphore_infos, last.signal_semaphore_info_count as _)
					.into_iter()
					.copied()
					.collect_in(a);
			signals.extend(signal);
			last.p_signal_semaphore_infos = signals.as_ptr();
			last.signal_semaphore_info_count += 1;

			self.device.queue_submit2(*queue, &submits, vk::Fence::null())?;
		}

		a.reset();
		Ok(signal.map(|x| SyncPoint { value: x.value }))
	}
}

pub struct Queue {
	pub family: u32,
	pub queue: Mutex<(vk::Queue, u64, Arena)>,
	pub semaphore: vk::Semaphore,
}

impl Queue {
	pub fn new(device: &ash::Device, family: u32) -> Result<Self> {
		unsafe {
			let queue = Mutex::new((
				device.get_device_queue2(
					&vk::DeviceQueueInfo2::builder()
						.queue_family_index(family)
						.queue_index(0),
				),
				0,
				Arena::with_block_size(1024),
			));
			let semaphore = device.create_semaphore(
				&vk::SemaphoreCreateInfo::builder().push_next(
					&mut vk::SemaphoreTypeCreateInfo::builder()
						.semaphore_type(vk::SemaphoreType::TIMELINE)
						.initial_value(0),
				),
				None,
			)?;

			Ok(Self {
				family,
				queue,
				semaphore,
			})
		}
	}

	pub fn destroy(&self, device: &ash::Device) {
		unsafe {
			device.destroy_semaphore(self.semaphore, None);
		}
	}
}

#[derive(Copy, Clone)]
pub struct Queues<T> {
	pub graphics: T,
	pub compute: T,
	pub transfer: T,
}

impl<T> Queues<T> {
	pub fn get(&self, ty: QueueType) -> &T {
		match ty {
			QueueType::Graphics => &self.graphics,
			QueueType::Compute => &self.compute,
			QueueType::Transfer => &self.transfer,
		}
	}

	pub fn get_mut(&mut self, ty: QueueType) -> &mut T {
		match ty {
			QueueType::Graphics => &mut self.graphics,
			QueueType::Compute => &mut self.compute,
			QueueType::Transfer => &mut self.transfer,
		}
	}

	pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Queues<U> {
		Queues {
			graphics: f(self.graphics),
			compute: f(self.compute),
			transfer: f(self.transfer),
		}
	}

	pub fn map_ref<U>(&self, mut f: impl FnMut(&T) -> U) -> Queues<U> {
		Queues {
			graphics: f(&self.graphics),
			compute: f(&self.compute),
			transfer: f(&self.transfer),
		}
	}

	pub fn map_mut<U>(&mut self, mut f: impl FnMut(&mut T) -> U) -> Queues<U> {
		Queues {
			graphics: f(&mut self.graphics),
			compute: f(&mut self.compute),
			transfer: f(&mut self.transfer),
		}
	}

	pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> std::result::Result<U, E>) -> std::result::Result<Queues<U>, E> {
		Ok(Queues {
			graphics: f(self.graphics)?,
			compute: f(self.compute)?,
			transfer: f(self.transfer)?,
		})
	}

	pub fn try_map_ref<U, E>(
		&self, mut f: impl FnMut(&T) -> std::result::Result<U, E>,
	) -> std::result::Result<Queues<U>, E> {
		Ok(Queues {
			graphics: f(&self.graphics)?,
			compute: f(&self.compute)?,
			transfer: f(&self.transfer)?,
		})
	}
}
