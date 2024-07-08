use std::{
	alloc::Allocator,
	fmt::Debug,
	iter,
	marker::PhantomData,
	sync::{
		atomic::{AtomicU64, Ordering},
		Mutex,
		MutexGuard,
	},
};

use ash::vk;
use tracing::{span, Level};

use crate::{
	arena::{IteratorAlloc, ToOwnedAlloc},
	device::Device,
	Result,
};

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Graphics;
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Compute;
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Transfer;
pub trait QueueType: sealed::Sealed {
	fn get<T>(q: &Queues<T>) -> &T;

	fn get_mut<T>(q: &mut Queues<T>) -> &mut T;

	fn into<T>(q: Queues<T>) -> T;
}
mod sealed {
	use super::*;

	pub trait Sealed {}
	impl Sealed for Graphics {}
	impl QueueType for Graphics {
		fn get<T>(q: &Queues<T>) -> &T { &q.graphics }

		fn get_mut<T>(q: &mut Queues<T>) -> &mut T { &mut q.graphics }

		fn into<T>(q: Queues<T>) -> T { q.graphics }
	}
	impl Sealed for Compute {}
	impl QueueType for Compute {
		fn get<T>(q: &Queues<T>) -> &T { &q.compute }

		fn get_mut<T>(q: &mut Queues<T>) -> &mut T { &mut q.compute }

		fn into<T>(q: Queues<T>) -> T { q.compute }
	}
	impl Sealed for Transfer {}
	impl QueueType for Transfer {
		fn get<T>(q: &Queues<T>) -> &T { &q.transfer }

		fn get_mut<T>(q: &mut Queues<T>) -> &mut T { &mut q.transfer }

		fn into<T>(q: Queues<T>) -> T { q.transfer }
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct SyncPoint<T: QueueType>(u64, PhantomData<fn() -> T>);

impl<TY: QueueType> SyncPoint<TY> {
	pub fn wait(self, device: &Device) -> Result<()> {
		unsafe {
			device
				.device()
				.wait_semaphores(
					&vk::SemaphoreWaitInfo::default()
						.semaphores(&[device.queues.get::<TY>().semaphore])
						.values(&[self.0]),
					u64::MAX,
				)
				.map_err(Into::into)
		}
	}

	pub fn is_complete(self, device: &Device) -> Result<bool> {
		unsafe {
			let v = device
				.device
				.get_semaphore_counter_value(device.queues.get::<TY>().semaphore)?;
			Ok(v >= self.0)
		}
	}

	pub fn later(self, other: Self) -> Self { Self(self.0.max(other.0), PhantomData) }
}

pub struct Queues<T> {
	pub graphics: T, // Also supports presentation.
	pub compute: T,
	pub transfer: T,
}

impl<T> Queues<T> {
	pub fn get<TY: QueueType>(&self) -> &T { TY::get(self) }

	pub fn get_mut<TY: QueueType>(&mut self) -> &mut T { TY::get_mut(self) }

	pub fn into<TY: QueueType>(self) -> T { TY::into(self) }

	pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Queues<U> {
		let Queues {
			graphics,
			compute,
			transfer,
		} = self;
		Queues {
			graphics: f(graphics),
			compute: f(compute),
			transfer: f(transfer),
		}
	}

	pub fn map_ref<U>(&self, mut f: impl FnMut(&T) -> U) -> Queues<U> {
		let Queues {
			graphics,
			compute,
			transfer,
		} = self;
		Queues {
			graphics: f(graphics),
			compute: f(compute),
			transfer: f(transfer),
		}
	}

	pub fn map_mut<U>(&mut self, mut f: impl FnMut(&mut T) -> U) -> Queues<U> {
		let Queues {
			graphics,
			compute,
			transfer,
		} = self;
		Queues {
			graphics: f(graphics),
			compute: f(compute),
			transfer: f(transfer),
		}
	}

	pub fn try_map_ref<U, E>(
		&self, mut f: impl FnMut(&T) -> std::result::Result<U, E>,
	) -> std::result::Result<Queues<U>, E> {
		let Queues {
			graphics,
			compute,
			transfer,
		} = self;
		Ok(Queues {
			graphics: f(graphics)?,
			compute: f(compute)?,
			transfer: f(transfer)?,
		})
	}

	pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> std::result::Result<U, E>) -> std::result::Result<Queues<U>, E> {
		let Queues {
			graphics,
			compute,
			transfer,
		} = self;
		Ok(Queues {
			graphics: f(graphics)?,
			compute: f(compute)?,
			transfer: f(transfer)?,
		})
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct SyncStage<S> {
	pub point: S,
	pub stage: vk::PipelineStageFlags2,
}

impl<TY: Copy + QueueType> SyncStage<SyncPoint<TY>> {
	pub fn merge(&mut self, other: Self) {
		self.point = self.point.later(other.point);
		self.stage |= other.stage;
	}

	fn info(self, qs: &Queues<QueueData>) -> vk::SemaphoreSubmitInfo<'static> {
		vk::SemaphoreSubmitInfo::default()
			.semaphore(qs.get::<TY>().semaphore)
			.value(self.point.0)
			.stage_mask(self.stage)
	}
}

impl SyncStage<vk::Semaphore> {
	fn info(self) -> vk::SemaphoreSubmitInfo<'static> {
		vk::SemaphoreSubmitInfo::default()
			.semaphore(self.point)
			.stage_mask(self.stage)
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct QueueWait<'a> {
	pub graphics: Option<SyncStage<SyncPoint<Graphics>>>,
	pub compute: Option<SyncStage<SyncPoint<Compute>>>,
	pub transfer: Option<SyncStage<SyncPoint<Transfer>>>,
	pub binary_semaphores: &'a [SyncStage<vk::Semaphore>],
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct QueueWaitOwned<A: Allocator> {
	pub graphics: Option<SyncStage<SyncPoint<Graphics>>>,
	pub compute: Option<SyncStage<SyncPoint<Compute>>>,
	pub transfer: Option<SyncStage<SyncPoint<Transfer>>>,
	pub binary_semaphores: Vec<SyncStage<vk::Semaphore>, A>,
}

impl<A: Allocator> Debug for QueueWaitOwned<A> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("QueueWaitOwned")
			.field("graphics", &self.graphics)
			.field("compute", &self.compute)
			.field("transfer", &self.transfer)
			.field("binary_semaphores", &self.binary_semaphores)
			.finish()
	}
}

impl<A: Allocator> QueueWaitOwned<A> {
	pub fn default(alloc: A) -> Self {
		Self {
			graphics: None,
			compute: None,
			transfer: None,
			binary_semaphores: Vec::new_in(alloc),
		}
	}

	pub fn clear(&mut self) {
		self.graphics = None;
		self.compute = None;
		self.transfer = None;
		self.binary_semaphores.clear();
	}

	pub fn borrow(&self) -> QueueWait {
		QueueWait {
			graphics: self.graphics,
			compute: self.compute,
			transfer: self.transfer,
			binary_semaphores: &self.binary_semaphores,
		}
	}

	pub fn is_empty(&self) -> bool {
		self.graphics.is_none()
			&& self.compute.is_none()
			&& self.transfer.is_none()
			&& self.binary_semaphores.is_empty()
	}

	pub fn merge(&mut self, other: Self) {
		match &mut self.graphics {
			Some(g) => {
				other.graphics.map(|x| g.merge(x));
			},
			None => self.graphics = other.graphics,
		}
		match &mut self.compute {
			Some(g) => {
				other.compute.map(|x| g.merge(x));
			},
			None => self.compute = other.compute,
		}
		match &mut self.transfer {
			Some(g) => {
				other.transfer.map(|x| g.merge(x));
			},
			None => self.transfer = other.transfer,
		}
		self.binary_semaphores.extend(other.binary_semaphores);
	}
}

impl<A: Allocator> ToOwnedAlloc<A> for QueueWait<'_> {
	type Owned = QueueWaitOwned<A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned {
		Self::Owned {
			graphics: self.graphics,
			compute: self.compute,
			transfer: self.transfer,
			binary_semaphores: self.binary_semaphores.to_owned_alloc(alloc),
		}
	}
}

pub struct QueueData {
	queue: Mutex<vk::Queue>,
	family: u32,
	semaphore: vk::Semaphore,
	value: AtomicU64,
}

impl QueueData {
	pub fn new(device: &ash::Device, family: u32) -> Result<Self> {
		unsafe {
			let queue = Mutex::new(device.get_device_queue(family, 0));
			let semaphore = device.create_semaphore(
				&vk::SemaphoreCreateInfo::default().push_next(
					&mut vk::SemaphoreTypeCreateInfo::default()
						.semaphore_type(vk::SemaphoreType::TIMELINE)
						.initial_value(0),
				),
				None,
			)?;

			Ok(Self {
				queue,
				family,
				semaphore,
				value: AtomicU64::new(0),
			})
		}
	}

	pub fn family(&self) -> u32 { self.family }

	pub fn queue(&self) -> MutexGuard<'_, vk::Queue> { self.queue.lock().unwrap() }

	pub fn current<T: QueueType>(&self) -> SyncPoint<T> { SyncPoint(self.value.load(Ordering::Acquire), PhantomData) }

	pub fn submit<T: QueueType>(
		&self, qs: &Queues<Self>, device: &Device, wait: QueueWait, bufs: &[vk::CommandBuffer],
		signal: &[SyncStage<vk::Semaphore>], fence: vk::Fence,
	) -> Result<SyncPoint<T>> {
		let s = span!(Level::TRACE, "gpu submit");
		let _e = s.enter();

		let wait: Vec<_, _> = wait
			.graphics
			.into_iter()
			.map(|x| x.info(qs))
			.chain(wait.compute.into_iter().map(|x| x.info(qs)))
			.chain(wait.transfer.into_iter().map(|x| x.info(qs)))
			.chain(wait.binary_semaphores.into_iter().map(|x| x.info()))
			.collect_in(device.arena());
		let infos: Vec<_, _> = bufs
			.iter()
			.map(|&b| vk::CommandBufferSubmitInfo::default().command_buffer(b))
			.collect_in(device.arena());
		let v = self.value.fetch_add(1, Ordering::Release);
		let signal: Vec<_, _> = iter::once(
			vk::SemaphoreSubmitInfo::default()
				.semaphore(self.semaphore)
				.value(v + 1)
				.stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
		)
		.chain(signal.into_iter().map(|x| x.info()))
		.collect_in(device.arena());

		unsafe {
			let s = span!(Level::TRACE, "driver submit");
			let _e = s.enter();
			let q = self.queue.lock().unwrap();
			device.device().queue_submit2(
				*q,
				&[vk::SubmitInfo2::default()
					.wait_semaphore_infos(&wait)
					.command_buffer_infos(&infos)
					.signal_semaphore_infos(&signal)],
				fence,
			)?;

			Ok(SyncPoint(v + 1, PhantomData))
		}
	}

	pub fn destroy(&self, device: &ash::Device) {
		unsafe {
			device.destroy_semaphore(self.semaphore, None);
		}
	}
}
