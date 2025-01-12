// https://github.com/h3r2tic/tony-mc-mapface/

use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{Frame, ImageDesc, ImageUsage, Res},
	resource::ImageView,
	sync::Shader,
	util::render::FullscreenPass,
	Result,
};
use vek::Vec3;

use crate::assets::image::{ImageAsset, ImageAssetView};

pub struct TonyMcMapfaceTonemap {
	lut: ImageAssetView,
	sampler: SamplerId,
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	input: ImageId,
	lut: ImageId,
	sampler: SamplerId,
	exp: f32,
}

impl TonyMcMapfaceTonemap {
	const LUT: &[u8] = include_bytes!("tony_mc_mapface.lut");

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			lut: ImageAssetView::new(
				"tony mcmapface lut",
				ImageAsset {
					size: Vec3::broadcast(48),
					format: vk::Format::E5B9G9R9_UFLOAT_PACK32.as_raw(),
					data: Self::LUT.into(),
				},
			)
			.unwrap(),
			sampler: device.sampler(SamplerDesc {
				address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				..Default::default()
			}),
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.tony_mc_mapface.main",
					spec: &[],
				},
				&[vk::Format::R8G8B8A8_SRGB],
			)?,
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, exp: f32) -> Res<ImageView> {
		let mut pass = frame.pass("tony mcmapface tonemap");

		pass.reference(input, ImageUsage::sampled_2d(Shader::Fragment));
		let desc = pass.desc(input);
		let out = pass.resource(
			ImageDesc {
				format: vk::Format::R8G8B8A8_SRGB,
				..desc
			},
			ImageUsage::color_attachment(),
		);

		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			self.pass.run_one(
				&mut pass,
				&PushConstants {
					input,
					lut: self.lut.view().id.unwrap(),
					sampler: self.sampler,
					exp,
				},
				out,
			)
		});

		out
	}

	// TODO: delete LUT
	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
