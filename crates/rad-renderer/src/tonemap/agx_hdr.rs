use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferUsage, Frame, ImageDesc, ImageUsage, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView},
	util::render::FullscreenPass,
	Result,
};

use crate::tonemap::agx::AgXLook;

pub struct AgxHdrTonemap {
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	exp: GpuPtr<f32>,
	input: ImageId,
	_pad: u32,
	look: AgXLook,
}

impl AgxHdrTonemap {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.agx_hdr.main",
					spec: &[],
				},
				&[vk::Format::A2B10G10R10_UNORM_PACK32],
			)?,
		})
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, exp: Res<BufferHandle>, look: AgXLook,
	) -> Res<ImageView> {
		let mut pass = frame.pass("agx hdr tonemap");

		pass.reference(input, ImageUsage::sampled_2d(Shader::Fragment));
		pass.reference(exp, BufferUsage::read(Shader::Fragment));
		let desc = pass.desc(input);
		let out = pass.resource(
			ImageDesc {
				format: vk::Format::A2B10G10R10_UNORM_PACK32,
				..desc
			},
			ImageUsage::color_attachment(),
		);

		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let exp = pass.get(exp).ptr();
			self.pass.run_one(
				&mut pass,
				&PushConstants {
					exp,
					input,
					_pad: 0,
					look,
				},
				out,
			);
		});

		out
	}

	pub unsafe fn destroy(self) { unsafe { self.pass.destroy(); }}
}
