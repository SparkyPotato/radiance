use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use rad_graph::{
	device::{descriptor::StorageImageId, Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, Frame, ImageDesc, ImageUsage, ImageUsageType, Res},
	resource::{GpuPtr, ImageView, Subresource},
	sync::Shader,
	util::render::FullscreenPass,
	Result,
};
use rand::{thread_rng, RngCore};
use vek::Vec2;

use crate::{mesh::CameraData, scene::GpuInstance, PrimaryViewData};

pub struct PathTracer {
	pass: FullscreenPass<PushConstants>,
}

#[derive(Clone)]
pub struct RenderInfo {
	pub data: PrimaryViewData,
	pub size: Vec2<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	out: StorageImageId,
	seed: u32,
	as_: GpuPtr<u8>,
}

impl PathTracer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.pt.main",
					spec: &[],
				},
				&[],
			)?,
		})
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> Res<ImageView> {
		let mut pass = frame.pass("path trace");

		let usage = BufferUsage {
			usages: &[BufferUsageType::ShaderStorageRead(Shader::Fragment)],
		};
		pass.reference(info.data.scene.as_, usage);
		pass.reference(info.data.scene.instances, usage);
		let camera = pass.resource(
			BufferDesc {
				size: std::mem::size_of::<CameraData>() as _,
				loc: BufferLoc::Upload,
				persist: None,
			},
			usage,
		);

		let out = pass.resource(
			ImageDesc {
				format: vk::Format::B10G11R11_UFLOAT_PACK32,
				size: vk::Extent3D {
					width: info.size.x,
					height: info.size.y,
					depth: 1,
				},
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				persist: None,
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[
					ImageUsageType::ShaderStorageRead(Shader::Fragment),
					ImageUsageType::ShaderStorageWrite(Shader::Fragment),
				],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		pass.build(move |mut pass| {
			let out = pass.get(out);
			let as_ = pass
				.get(info.data.scene.as_)
				.ptr()
				.offset(info.data.scene.as_offset as _);
			let instances = pass.get(info.data.scene.instances).ptr();
			let mut camera = pass.get(camera);
			unsafe { camera.data.as_mut() }
				.write(bytes_of(&CameraData::new(
					info.size.x as f32 / info.size.y as f32,
					info.data.camera,
					info.data.transform,
				)))
				.unwrap();
			self.pass.run_empty(
				&mut pass,
				&PushConstants {
					instances,
					camera: camera.ptr(),
					out: out.storage_id.unwrap(),
					seed: thread_rng().next_u32(),
					as_,
				},
				vk::Extent2D::default().width(out.size.width).height(out.size.height),
			);
		});

		out
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
