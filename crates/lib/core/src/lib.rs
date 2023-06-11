use std::ops::Deref;

use radiance_graph::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{ExecutionSnapshot, Frame, RenderGraph},
	Result,
};
use radiance_shader_compiler::runtime::{ShaderBlob, ShaderRuntime};
use radiance_util::{
	deletion::DeletionQueue,
	pipeline::PipelineCache,
	staging::{StageTicket, Staging, StagingCtx},
};

pub struct CoreDevice {
	pub device: Device,
	pub arena: Arena,
}

impl CoreDevice {
	pub fn new(device: Device) -> Result<Self> {
		let arena = Arena::new();
		Ok(Self { device, arena })
	}
}

impl Deref for CoreDevice {
	type Target = Device;

	fn deref(&self) -> &Self::Target { &self.device }
}

pub type CoreFrame<'pass, 'graph> = Frame<'pass, 'graph, &'graph mut RenderCore>;

pub struct RenderCore {
	pub shaders: ShaderRuntime,
	pub cache: PipelineCache,
	pub staging: Staging,
	pub delete: DeletionQueue,
	last_frame_snapshot: ExecutionSnapshot,
}

impl RenderCore {
	pub fn new<'s>(device: &CoreDevice, shaders: impl IntoIterator<Item = &'s ShaderBlob>) -> Result<Self> {
		let shaders = ShaderRuntime::new(device.device(), shaders);
		let cache = PipelineCache::new(&device)?;
		let staging = Staging::new(&device)?;
		Ok(Self {
			shaders,
			last_frame_snapshot: ExecutionSnapshot::default(),
			cache,
			staging,
			delete: DeletionQueue::new(),
		})
	}

	pub fn frame<'pass, 'graph>(
		&'passmut self, device: &'graph CoreDevice, graph: &'graph mut RenderGraph,
	) -> Result<CoreFrame<'pass, 'graph>> {
		self.staging.poll(device)?;
		let snap = graph.snapshot();
		self.delete.advance(device);
		self.last_frame_snapshot = snap;
		let frame = graph.frame(&device.arena, self);
		Ok(frame)
	}

	pub fn stage(
		&mut self, device: &CoreDevice, exec: impl FnOnce(&mut StagingCtx) -> Result<()>,
	) -> Result<StageTicket> {
		self.staging.stage(
			device,
			self.last_frame_snapshot
				.as_submit_info()
				.into_iter()
				.collect_in(&device.arena),
			exec,
		)
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		self.delete.destroy(device);
		self.staging.destroy(device);
		self.cache.destroy(device);
		self.shaders.destroy(device.device());
	}
}
