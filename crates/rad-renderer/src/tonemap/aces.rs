use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferUsage, Frame, ImageDesc, ImageUsage, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView},
	util::{
		pass::{Attachment, Load},
		render::FullscreenPass,
	},
	Result,
};

pub struct AcesTonemap {
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	input: ImageId,
	_pad: u32,
	exp: GpuPtr<f32>,
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

	/// `highlights` must be sorted.
	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, exp: Res<BufferHandle>,
	) -> Res<ImageView> {
		let mut pass = frame.pass("aces tonemap");

		pass.reference(input, ImageUsage::sampled_2d(Shader::Fragment));
		pass.reference(exp, BufferUsage::read(Shader::Fragment));
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
			let exp = pass.get(exp).ptr();
			self.pass.run(
				&mut pass,
				&PushConstants { input, _pad: 0, exp },
				&[Attachment {
					image: out,
					load: Load::Clear(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					}),
					store: true,
				}],
			)
		});

		out
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
