use std::{any::Any, sync::Arc};

use bytemuck::NoUninit;
use radiance_graph::{
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, ExternalBuffer, Frame, Res},
	resource::{BufferHandle, GpuPtr},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};

use crate::{
	io::SliceWriter,
	scene::{GpuInstance, Transform},
	AssetRuntime,
};

pub struct SceneRuntime {
	tick: ComputePass<PushConstants>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct TransformUpdate {
	pub instance: u32,
	pub transform: Transform,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	updates: GpuPtr<TransformUpdate>,
	frame: u64,
	count: u32,
	_pad: u32,
}

impl AssetRuntime for SceneRuntime {
	fn initialize(device: &Device) -> Result<Self>
	where
		Self: Sized,
	{
		Ok(Self {
			tick: ComputePass::new(
				device,
				ShaderInfo {
					shader: "asset.scene.tick",
					spec: &[],
				},
			)?,
		})
	}

	fn as_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> { self }
}

impl SceneRuntime {
	pub fn tick<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, instances: BufferHandle, frame_index: u64,
		updates: impl ExactSizeIterator<Item = TransformUpdate> + 'pass,
	) -> Res<BufferHandle> {
		let mut pass = frame.pass("update scene");
		let count = updates.len();
		let instances = pass.resource(
			ExternalBuffer { handle: instances },
			BufferUsage {
				usages: if count > 0 {
					&[
						BufferUsageType::ShaderStorageRead(Shader::Compute),
						BufferUsageType::ShaderStorageWrite(Shader::Compute),
					]
				} else {
					&[]
				},
			},
		);
		let update_buffer = (count > 0).then(|| {
			pass.resource(
				BufferDesc {
					size: (count * std::mem::size_of::<TransformUpdate>()) as _,
					loc: BufferLoc::Upload,
					persist: None,
				},
				BufferUsage {
					usages: &[BufferUsageType::ShaderStorageRead(Shader::Compute)],
				},
			)
		});
		let count = count as _;
		pass.build(move |mut pass| unsafe {
			if let Some(update_buffer) = update_buffer {
				let mut update_buf = pass.get(update_buffer);
				let mut w = SliceWriter::new(update_buf.data.as_mut());
				for u in updates {
					w.write(u).unwrap();
				}
				self.tick.dispatch(
					&PushConstants {
						instances: pass.get(instances).ptr(),
						updates: update_buf.ptr(),
						frame: frame_index,
						count,
						_pad: 0,
					},
					&pass,
					(count + 63) / 64,
					1,
					1,
				);
			}
		});
		instances
	}
}
