use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, Frame, ImageUsage, ImageUsageType, Res},
	resource::{BufferHandle, GpuPtr, ImageView, Subresource},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	histogram: GpuPtr<u32>,
	exp: GpuPtr<f32>,
	input: ImageId,
	min_exp: f32,
	inv_exp_range: f32,
	lerp_coeff: f32,
}

pub struct ExposureCalc {
	histogram: ComputePass<PushConstants>,
	average: ComputePass<PushConstants>,
	read_histogram: [u32; 256],
	exposure: f32,
}

pub struct ExposureStats {
	pub exposure: f32,
	pub histogram: [u32; 256],
}

impl ExposureCalc {
	pub const MAX_EXPOSURE: f32 = 17.0;
	pub const MIN_EXPOSURE: f32 = -6.0;

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			histogram: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.exposure.histogram",
					spec: &[],
				},
			)?,
			average: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.exposure.average",
					spec: &[],
				},
			)?,
			read_histogram: [0; 256],
			exposure: 0.0,
		})
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, dt: f32,
	) -> (Res<BufferHandle>, ExposureStats) {
		let Self {
			histogram: hist,
			average,
			read_histogram,
			exposure,
		} = self;

		let histogram_size = std::mem::size_of::<u32>() as u64 * 256;
		let exp_size = std::mem::size_of::<f32>() as u64;

		let mut pass = frame.pass("zero histogram");
		let histogram = pass.resource(
			BufferDesc {
				size: histogram_size,
				loc: BufferLoc::GpuOnly,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		let exp = pass.resource(
			BufferDesc {
				size: exp_size,
				loc: BufferLoc::GpuOnly,
				persist: Some("exposure"),
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		pass.build(move |mut pass| {
			pass.zero(histogram);
			pass.zero_if_uninit(exp);
		});

		let inp_usage = ImageUsage {
			format: vk::Format::UNDEFINED,
			usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Compute)],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};

		let mut pass = frame.pass("generate histogram");
		pass.reference(input, inp_usage);
		pass.reference(
			histogram,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);

		let size = pass.desc(input).size;
		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let histogram = pass.get(histogram).ptr();
			hist.dispatch(
				&mut pass,
				&PushConstants {
					histogram,
					exp: GpuPtr::null(),
					input,
					lerp_coeff: 0.0,
					min_exp: Self::MIN_EXPOSURE,
					inv_exp_range: 1.0 / (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE),
				},
				(size.width + 15) >> 4,
				(size.height + 15) >> 4,
				1,
			)
		});

		let mut pass = frame.pass("calculate exposure");
		pass.reference(input, inp_usage);
		pass.reference(
			histogram,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Compute)],
			},
		);
		pass.reference(
			exp,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);

		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let histogram = pass.get(histogram).ptr();
			let exp = pass.get(exp).ptr();
			average.dispatch(
				&mut pass,
				&PushConstants {
					histogram,
					exp,
					input,
					min_exp: Self::MIN_EXPOSURE,
					inv_exp_range: 1.0 / (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE),
					lerp_coeff: (1.0 - (-dt).exp()).clamp(0.0, 1.0),
				},
				1,
				1,
				1,
			);
		});

		let mut pass = frame.pass("readback exposure");
		let usage = BufferUsage {
			usages: &[BufferUsageType::TransferWrite],
		};
		let histogram_read = pass.resource(
			BufferDesc {
				size: histogram_size,
				loc: BufferLoc::Readback,
				persist: Some("histogram readback"),
			},
			usage,
		);
		let exp_read = pass.resource(
			BufferDesc {
				size: exp_size,
				loc: BufferLoc::Readback,
				persist: Some("exposure readback"),
			},
			usage,
		);
		let usage = BufferUsage {
			usages: &[BufferUsageType::TransferRead],
		};
		pass.reference(histogram, usage);
		pass.reference(exp, usage);

		let ret_exp = *exposure;
		let hist = *read_histogram;
		pass.build(move |mut pass| {
			pass.copy_full_buffer(histogram, histogram_read, 0);
			*read_histogram = pass.readback(histogram_read, 0);

			pass.copy_full_buffer(exp, exp_read, 0);
			*exposure = pass.readback(exp_read, 0);
		});

		(
			exp,
			ExposureStats {
				exposure: ret_exp,
				histogram: hist,
			},
		)
	}

	pub unsafe fn destroy(self) {
		self.histogram.destroy();
		self.average.destroy();
	}
}
