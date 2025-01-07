use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{Frame, ImageDesc, ImageUsage, Res, Shader},
	resource::ImageView,
	util::render::FullscreenPass,
	Result,
};

pub struct AcesTonemap {
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	input: ImageId,
	exp: f32,
}

impl AcesTonemap {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.aces.main",
					spec: &[],
				},
				&[vk::Format::R8G8B8A8_SRGB],
			)?,
		})
	}

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, exp: f32) -> Res<ImageView> {
		let mut pass = frame.pass("aces tonemap");

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
			self.pass.run_one(&mut pass, &PushConstants { input, exp }, out)
		});

		out
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
