use ash::vk;

use crate::{
	cmd::CommandPool,
	device::{Compute, Device, Graphics, QueueSyncs, QueueType, Queues, SyncPoint, Transfer},
	graph::{Deletable, Resource},
	Error,
	Result,
};

pub struct AsyncExecutor {
	pools: Queues<CommandPool>,
	syncs: QueueSyncs,
	deletes: Queues<Vec<Resource>>,
}

pub struct AsyncCtx<'a> {
	exec: &'a mut AsyncExecutor,
	bufs: Queues<vk::CommandBuffer>,
	used: Queues<bool>,
}

impl AsyncExecutor {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pools: device.queue_families().try_map(|x| CommandPool::new(device, x))?,
			syncs: QueueSyncs::default(),
			deletes: Default::default(),
		})
	}

	pub fn start(&mut self, device: &Device) -> Result<AsyncCtx> {
		Ok(AsyncCtx {
			bufs: self.pools.try_map_mut(|x| unsafe {
				let b = x.next(device)?;
				device.device().begin_command_buffer(
					b,
					&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)?;
				Ok::<_, Error>(b)
			})?,
			used: Default::default(),
			exec: self,
		})
	}

	pub fn tick(&mut self, device: &Device) -> Result<()> {
		unsafe {
			if let Some(g) = self.syncs.graphics {
				if g.is_complete(device)? {
					self.pools.graphics.reset(device)?;
					for r in self.deletes.graphics.drain(..) {
						r.destroy(device);
					}
				}
			}
			if let Some(g) = self.syncs.compute {
				if g.is_complete(device)? {
					self.pools.compute.reset(device)?;
					for r in self.deletes.compute.drain(..) {
						r.destroy(device);
					}
				}
			}
			if let Some(g) = self.syncs.transfer {
				if g.is_complete(device)? {
					self.pools.transfer.reset(device)?;
					for r in self.deletes.transfer.drain(..) {
						r.destroy(device);
					}
				}
			}

			Ok(())
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pools.map(|x| x.destroy(device));
		self.deletes.map(|x| {
			for r in x {
				r.destroy(device)
			}
		});
	}
}

impl AsyncCtx<'_> {
	pub fn get_buf<TY: QueueType>(&mut self) -> vk::CommandBuffer {
		*self.used.get_mut::<TY>() = true;
		*self.bufs.get::<TY>()
	}

	pub fn delete<TY: QueueType>(&mut self, obj: impl Deletable) {
		obj.into_resources(self.exec.deletes.get_mut::<TY>());
	}

	pub fn finish(mut self, device: &Device) -> Result<QueueSyncs> {
		let graphics = self.inner::<Graphics>(device)?;
		let compute = self.inner::<Compute>(device)?;
		let transfer = self.inner::<Transfer>(device)?;
		let ret = QueueSyncs {
			graphics,
			compute,
			transfer,
		};
		self.exec.syncs.merge(ret);
		Ok(ret)
	}

	fn inner<TY: QueueType>(&mut self, device: &Device) -> Result<Option<SyncPoint<TY>>> {
		Ok(if !self.used.get::<TY>() {
			unsafe {
				self.exec.pools.get_mut::<TY>().reset(device)?;
			}
			None
		} else {
			unsafe {
				device.device().end_command_buffer(*self.bufs.get::<TY>())?;
			}
			Some(device.submit::<TY>(
				Default::default(),
				&[*self.bufs.get::<TY>()],
				Default::default(),
				Default::default(),
			)?)
		})
	}
}
