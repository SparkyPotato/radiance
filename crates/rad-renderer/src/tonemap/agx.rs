use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferUsage, Frame, ImageDesc, ImageUsage, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView},
	util::render::FullscreenPass,
	Result,
};
use vek::Vec3;

pub struct AgXTonemap {
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
pub struct AgXLook {
	pub offset: Vec3<f32>,
	pub slope: Vec3<f32>,
	pub power: Vec3<f32>,
	pub sat: f32,
}

impl Default for AgXLook {
	fn default() -> Self {
		Self {
			offset: Vec3::broadcast(0.0),
			slope: Vec3::broadcast(1.0),
			power: Vec3::broadcast(1.0),
			sat: 1.0,
		}
	}
}

impl AgXLook {
	pub fn punchy() -> Self {
		Self {
			offset: Vec3::broadcast(0.0),
			slope: Vec3::broadcast(1.0),
			power: Vec3::broadcast(1.1),
			sat: 1.1,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	exp: GpuPtr<f32>,
	input: ImageId,
	_pad: u32,
	look: AgXLook,
}

impl AgXTonemap {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.agx.main",
					spec: &[],
				},
				&[vk::Format::R8G8B8A8_SRGB],
			)?,
		})
	}

	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, exp: Res<BufferHandle>, look: AgXLook,
	) -> Res<ImageView> {
		let mut pass = frame.pass("agx tonemap");

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
			self.pass.run_one(
				&mut pass,
				&PushConstants {
					exp,
					input,
					_pad: 0,
					look,
				},
				out,
			)
		});

		out
	}

	pub unsafe fn destroy(self) { unsafe { self.pass.destroy(); }}
}
