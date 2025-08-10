use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	Result,
	device::{
		Device,
		SamplerDesc,
		ShaderInfo,
		descriptor::{ImageId, SamplerId},
	},
	graph::{Frame, ImageDesc, ImageUsage, PassBuilder, PassContext, Persist, Res},
	resource::ImageView,
	sync::Shader,
	util::{
		pass::{Attachment, Load},
		render::FullscreenPass,
	},
};
use vek::Vec3;

use crate::scene::{WorldRenderer, camera::CameraScene};

pub struct SkyLuts {
	transmittance: FullscreenPass<()>,
	scattering: FullscreenPass<ScatteringConstants>,
	eval: FullscreenPass<EvalConstants>,
	transmittance_image: Persist<ImageView>,
	scattering_image: Persist<ImageView>,
	sampler: SamplerId,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuSkySampler {
	lut: ImageId,
	transmittance: ImageId,
	sampler: SamplerId,
	sun_dir: Vec3<f32>,
	sun_radiance: Vec3<f32>,
}

#[derive(Copy, Clone)]
pub struct SkySampler {
	pub lut: Res<ImageView>,
	transmittance: Res<ImageView>,
	sampler: SamplerId,
	sun_dir: Vec3<f32>,
	sun_radiance: Vec3<f32>,
}

impl SkySampler {
	pub fn reference(&self, pass: &mut PassBuilder, shader: Shader) {
		pass.reference(self.lut, ImageUsage::sampled_2d(shader));
		pass.reference(self.transmittance, ImageUsage::sampled_2d(shader));
	}

	pub fn to_gpu(&self, pass: &mut PassContext) -> GpuSkySampler {
		GpuSkySampler {
			lut: pass.get(self.lut).id.unwrap(),
			transmittance: pass.get(self.transmittance).id.unwrap(),
			sampler: self.sampler,
			sun_dir: self.sun_dir,
			sun_radiance: self.sun_radiance,
		}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct ScatteringConstants {
	transmittance: ImageId,
	sampler: SamplerId,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct EvalConstants {
	transmittance: ImageId,
	scattering: ImageId,
	sampler: SamplerId,
	cam_pos: Vec3<f32>,
	sun_dir: Vec3<f32>,
}

impl SkyLuts {
	const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			transmittance: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.sky.transmittance.main",
					spec: &[],
				},
				&[Self::FORMAT],
			)?,
			scattering: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.sky.scattering.main",
					spec: &[],
				},
				&[Self::FORMAT],
			)?,
			eval: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.sky.eval.main",
					spec: &[],
				},
				&[Self::FORMAT],
			)?,
			transmittance_image: Persist::new(),
			scattering_image: Persist::new(),
			sampler: device.sampler(SamplerDesc::default()),
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, rend: &mut WorldRenderer<'pass, '_>) -> SkySampler {
		let camera = rend.get::<CameraScene>(frame);

		frame.start_region("sky");
		let format = Self::FORMAT;

		let mut pass = frame.pass("transmittance");
		let trans = pass.resource(
			ImageDesc {
				size: vk::Extent3D {
					width: 256,
					height: 64,
					depth: 1,
				},
				format,
				persist: Some(self.transmittance_image),
				..Default::default()
			},
			ImageUsage::color_attachment(),
		);
		pass.build(move |mut pass| {
			if pass.is_uninit(trans) {
				self.transmittance.run(
					&mut pass,
					&(),
					&[Attachment {
						image: trans,
						load: Load::DontCare,
						store: true,
					}],
				);
			}
		});

		let mut pass = frame.pass("scattering");
		pass.reference(trans, ImageUsage::sampled_2d(Shader::Fragment));
		let scatter = pass.resource(
			ImageDesc {
				size: vk::Extent3D {
					width: 32,
					height: 32,
					depth: 1,
				},
				format,
				persist: Some(self.scattering_image),
				..Default::default()
			},
			ImageUsage::color_attachment(),
		);
		pass.build(move |mut pass| {
			if pass.is_uninit(scatter) {
				let transmittance = pass.get(trans).id.unwrap();
				self.scattering.run(
					&mut pass,
					&ScatteringConstants {
						transmittance,
						sampler: self.sampler,
					},
					&[Attachment {
						image: scatter,
						load: Load::DontCare,
						store: true,
					}],
				);
			}
		});

		let mut pass = frame.pass("eval");
		pass.reference(trans, ImageUsage::sampled_2d(Shader::Fragment));
		pass.reference(scatter, ImageUsage::sampled_2d(Shader::Fragment));
		let lut = pass.resource(
			ImageDesc {
				size: vk::Extent3D {
					width: 256,
					height: 192,
					depth: 1,
				},
				format,
				..Default::default()
			},
			ImageUsage::color_attachment(),
		);
		pass.build(move |mut pass| {
			let transmittance = pass.get(trans).id.unwrap();
			let scattering = pass.get(scatter).id.unwrap();
			self.eval.run(
				&mut pass,
				&EvalConstants {
					transmittance,
					scattering,
					sampler: self.sampler,
					cam_pos: camera.curr.transform.position,
					// TODO: fix
					sun_dir: Vec3::zero(),
				},
				&[Attachment {
					image: lut,
					load: Load::DontCare,
					store: true,
				}],
			);
		});

		frame.end_region();

		SkySampler {
			lut,
			transmittance: trans,
			sampler: self.sampler,
			// TODO: fix
			sun_dir: Vec3::zero(),
			sun_radiance: Vec3::zero(),
		}
	}

	pub unsafe fn destroy(self) {
		unsafe {
			self.transmittance.destroy();
			self.scattering.destroy();
			self.eval.destroy();
		}
	}
}
