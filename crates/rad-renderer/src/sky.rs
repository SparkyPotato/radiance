use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{Frame, ImageDesc, ImageUsage, PassBuilder, PassContext, Res},
	resource::ImageView,
	sync::Shader,
	util::{
		pass::{Attachment, Load},
		render::FullscreenPass,
	},
	Result,
};
use vek::Vec3;

use crate::PrimaryViewData;

pub struct SkyLuts {
	transmittance: FullscreenPass<()>,
	scattering: FullscreenPass<ScatteringConstants>,
	eval: FullscreenPass<EvalConstants>,
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
	lut: Res<ImageView>,
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
			sampler: device.sampler(SamplerDesc::default()),
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, data: PrimaryViewData) -> SkySampler {
		frame.start_region("sky lut");
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
				..Default::default()
			},
			ImageUsage::color_attachment(),
		);
		pass.build(move |mut pass| {
			self.transmittance.run(
				&mut pass,
				&(),
				&[Attachment {
					image: trans,
					load: Load::DontCare,
					store: true,
				}],
			);
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
				..Default::default()
			},
			ImageUsage::color_attachment(),
		);
		pass.build(move |mut pass| {
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
		});

		let mut pass = frame.pass("eval");
		pass.reference(trans, ImageUsage::sampled_2d(Shader::Fragment));
		pass.reference(scatter, ImageUsage::sampled_2d(Shader::Fragment));
		let lut = pass.resource(
			ImageDesc {
				size: vk::Extent3D {
					width: 256,
					height: 256,
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
					cam_pos: data.transform.position,
					sun_dir: -data.scene.sun_dir,
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
			sun_dir: -data.scene.sun_dir,
			sun_radiance: data.scene.sun_radiance,
		}
	}
}
