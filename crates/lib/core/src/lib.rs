use std::ops::Deref;

use radiance_graph::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{ExecutionSnapshot, Frame, PassBuilder, PassContext, RenderGraph},
	Result,
};
use radiance_shader_compiler::runtime::{shader, ShaderBlob, ShaderRuntime};
use radiance_util::{
	deletion::DeletionQueue,
	pipeline::PipelineCache,
	staging::{StageError, StageTicket, Staging, StagingCtx},
};

pub mod persistent;
pub mod pipeline;

pub const SHADERS: ShaderBlob = shader!("radiance-core");

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

pub type CorePass<'frame, 'graph> = PassContext<'frame, 'graph, &'graph mut RenderCore>;

pub type CoreBuilder<'frame, 'pass, 'graph> = PassBuilder<'frame, 'pass, 'graph, &'graph mut RenderCore>;

pub struct RenderCore {
	pub shaders: ShaderRuntime,
	pub cache: PipelineCache,
	pub staging: Staging,
	pub delete: DeletionQueue,
	pub last_frame_snapshot: ExecutionSnapshot,
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
		&'graph mut self, device: &'graph CoreDevice, graph: &'graph mut RenderGraph,
	) -> Result<CoreFrame<'pass, 'graph>> {
		self.staging.poll(device)?;
		let snap = graph.snapshot();
		self.delete.advance(device);
		self.last_frame_snapshot = snap;
		let frame = graph.frame(&device.arena, self);
		Ok(frame)
	}

	pub fn stage<T, E>(
		&mut self, device: &CoreDevice,
		exec: impl FnOnce(&mut StagingCtx, &mut DeletionQueue) -> std::result::Result<T, E>,
	) -> std::result::Result<(T, StageTicket), StageError<E>> {
		self.staging.stage(
			device,
			self.last_frame_snapshot
				.as_submit_info()
				.into_iter()
				.collect_in(&device.arena),
			|ctx| exec(ctx, &mut self.delete),
		)
	}

	/// # Safety
	/// Appropriate synchronization must be performed before calling this function.
	pub unsafe fn destroy(self, device: &CoreDevice) {
		self.delete.destroy(device);
		self.staging.destroy(device);
		self.cache.destroy(device);
		self.shaders.destroy(device.device());
	}
}

pub trait PassBuilderExt {
	fn stage<T, E>(
		&mut self, device: &CoreDevice,
		exec: impl FnOnce(&mut StagingCtx, &mut DeletionQueue) -> std::result::Result<T, E>,
	) -> std::result::Result<T, StageError<E>>;
}

impl PassBuilderExt for CoreBuilder<'_, '_, '_> {
	fn stage<T, E>(
		&mut self, device: &CoreDevice,
		exec: impl FnOnce(&mut StagingCtx, &mut DeletionQueue) -> std::result::Result<T, E>,
	) -> std::result::Result<T, StageError<E>> {
		let (ret, ticket) = self.ctx().stage(device, exec)?;
		self.wait_on(ticket.as_info());
		Ok(ret)
	}
}
