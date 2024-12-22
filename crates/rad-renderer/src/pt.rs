use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit};
use rad_graph::{
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
		ShaderInfo,
	},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, Frame, ImageDesc, ImageUsage, ImageUsageType, Res},
	resource::{GpuPtr, ImageView, Subresource},
	sync::Shader,
	util::render::FullscreenPass,
	Result,
};
use rad_world::system::WorldId;
use rand::{thread_rng, RngCore};
use vek::Vec2;

use crate::{
	mesh::CameraData,
	scene::{GpuInstance, GpuLight, GpuTransform},
	PrimaryViewData,
};

// TODO: reset on world edit.
pub struct PathTracer {
	pass: FullscreenPass<PushConstants>,
	sampler: vk::Sampler,
	sampler_id: SamplerId,
	cached: Option<(WorldId, GpuTransform, Vec2<u32>)>,
	samples: u32,
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
	lights: GpuPtr<GpuLight>,
	camera: GpuPtr<CameraData>,
	as_: GpuPtr<u8>,
	sampler: SamplerId,
	out: StorageImageId,
	seed: u32,
	samples: u32,
	light_count: u32,
	sky_light: u32,
}

impl PathTracer {
	pub fn new(device: &Device) -> Result<Self> {
		// TODO: cringe
		let sampler = unsafe {
			device.device().create_sampler(
				&vk::SamplerCreateInfo::default()
					.mag_filter(vk::Filter::LINEAR)
					.min_filter(vk::Filter::LINEAR)
					.mipmap_mode(vk::SamplerMipmapMode::NEAREST)
					.address_mode_u(vk::SamplerAddressMode::REPEAT)
					.address_mode_v(vk::SamplerAddressMode::REPEAT)
					.address_mode_w(vk::SamplerAddressMode::REPEAT),
				None,
			)?
		};

		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.pt.main",
					spec: &[],
				},
				&[],
			)?,
			sampler,
			sampler_id: device.descriptors().get_sampler(device, sampler),
			cached: None,
			samples: 0,
		})
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, info: RenderInfo) -> (Res<ImageView>, u32) {
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
				format: vk::Format::R32G32B32A32_SFLOAT,
				size: vk::Extent3D {
					width: info.size.x,
					height: info.size.y,
					depth: 1,
				},
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				persist: Some("path tracer accum"),
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

		if let Some(c) = self.cached {
			if c.0 != info.data.id || c.1 != info.data.transform || c.2 != info.size {
				self.samples = 0;
			}
		}
		self.cached = Some((info.data.id, info.data.transform, info.size));

		let s = self.samples;
		pass.build(move |mut pass| {
			if pass.is_uninit(out) {
				self.samples = 0;
			}
			let out = pass.get(out);
			let as_ = pass
				.get(info.data.scene.as_)
				.ptr()
				.offset(info.data.scene.as_offset as _);
			let instances = pass.get(info.data.scene.instances).ptr();
			let lights = pass.get(info.data.scene.lights).ptr();
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
					lights,
					camera: camera.ptr(),
					as_,
					sampler: self.sampler_id,
					out: out.storage_id.unwrap(),
					seed: thread_rng().next_u32(),
					samples: self.samples,
					light_count: info.data.scene.light_count,
					sky_light: info.data.scene.sky_light,
				},
				vk::Extent2D::default().width(out.size.width).height(out.size.height),
			);
			self.samples += 1;
		});

		(out, s)
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pass.destroy();
		unsafe {
			device.device().destroy_sampler(self.sampler, None);
			device.descriptors().return_sampler(self.sampler_id);
		}
	}
}
