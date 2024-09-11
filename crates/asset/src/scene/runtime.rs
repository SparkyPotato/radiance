use std::{any::Any, sync::Arc};

use bytemuck::NoUninit;
use radiance_graph::{
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, Frame},
	resource::GpuPtr,
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
	update: ComputePass<PushConstants>,
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
	count: u32,
	_pad: u32,
}

impl AssetRuntime for SceneRuntime {
	fn initialize(device: &Device) -> Result<Self>
	where
		Self: Sized,
	{
		Ok(Self {
			update: ComputePass::new(
				device,
				ShaderInfo {
					shader: "asset.scene_update.main",
					spec: &[],
				},
			)?,
		})
	}

	fn as_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> { self }
}

impl SceneRuntime {
	pub fn update<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, instances: GpuPtr<GpuInstance>,
		updates: impl ExactSizeIterator<Item = TransformUpdate> + 'pass,
	) {
		if updates.len() == 0 {
			return;
		}

		let mut pass = frame.pass("update scene");
		let count = updates.len();
		let update_buffer = pass.resource(
			BufferDesc {
				size: (count * std::mem::size_of::<TransformUpdate>()) as _,
				loc: BufferLoc::Upload,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Compute)],
			},
		);
		let count = count as _;
		pass.build(move |mut pass| unsafe {
			let mut update_buf = pass.get(update_buffer);
			let mut w = SliceWriter::new(update_buf.data.as_mut());
			for u in updates {
				w.write(u).unwrap();
			}
			self.update.dispatch(
				&PushConstants {
					instances,
					updates: update_buf.ptr(),
					count,
					_pad: 0,
				},
				&pass,
				(count + 63) / 64,
				1,
				1,
			)
		});
	}
}
