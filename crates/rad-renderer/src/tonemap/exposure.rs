use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferDesc, BufferUsage, Frame, ImageUsage, Res},
	resource::{GpuPtr, ImageView},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	histogram: GpuPtr<u32>,
	input: ImageId,
	min_exp: f32,
	inv_exp_range: f32,
	lerp_coeff: f32,
}

pub struct ExposureCalc {
	histogram: ComputePass<PushConstants>,
	exposure: f32,
	target_exposure: f32,
	read_histogram: [u32; 256],
}

pub struct ExposureStats {
	pub exposure: f32,
	pub target_exposure: f32,
	pub histogram: [u32; 256],
}

impl ExposureCalc {
	pub const MAX_EXPOSURE: f32 = 18.0;
	pub const MAX_HISTOGRAM_RANGE: f32 = 0.9;
	pub const MIN_EXPOSURE: f32 = -6.0;
	pub const MIN_HISTOGRAM_RANGE: f32 = 0.6;

	pub fn bin_to_exposure(bin: f32) -> f32 {
		let log = (bin - 1.0) / 254.0;
		log * (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE) + Self::MIN_EXPOSURE
	}

	pub fn exposure_to_bin(exp: f32) -> f32 {
		(exp - Self::MIN_EXPOSURE) / (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE) * 254.0 + 1.0
	}

	pub fn exposure_to_lum(exp: f32) -> f32 { (exp - 3.0).exp2() }

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			histogram: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.exposure.histogram",
					spec: &[],
				},
			)?,
			exposure: 0.0,
			target_exposure: 0.0,
			read_histogram: [0; 256],
		})
	}

	pub fn run<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, dt: f32) -> ExposureStats {
		let Self {
			histogram: hist,
			exposure,
			target_exposure,
			read_histogram,
		} = self;

		let histogram_size = std::mem::size_of::<u32>() as u64 * 256;

		let mut pass = frame.pass("zero histogram");
		let histogram = pass.resource(BufferDesc::gpu(histogram_size), BufferUsage::transfer_write());
		pass.build(move |mut pass| pass.zero(histogram));

		let mut pass = frame.pass("generate histogram");
		pass.reference(input, ImageUsage::sampled_2d(Shader::Compute));
		pass.reference(histogram, BufferUsage::read_write(Shader::Compute));
		let size = pass.desc(input).size;
		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let histogram = pass.get(histogram).ptr();
			hist.dispatch(
				&mut pass,
				&PushConstants {
					histogram,
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

		let mut pass = frame.pass("readback histogram");
		let histogram_read = pass.resource(
			BufferDesc::readback(histogram_size, "histogram readback"),
			BufferUsage::transfer_write(),
		);
		pass.reference(histogram, BufferUsage::transfer_read());

		let ret_exp = *exposure;
		let target_exp = *target_exposure;
		let hist = *read_histogram;
		pass.build(move |mut pass| {
			pass.copy_full_buffer(histogram, histogram_read, 0);
			*read_histogram = pass.readback(histogram_read, 0);

			let total: u32 = read_histogram.iter().skip(1).sum();
			let range_start = total as f32 * Self::MIN_HISTOGRAM_RANGE;
			let range_end = total as f32 * Self::MAX_HISTOGRAM_RANGE;
			let mut seen = 0.0;
			let mut sum = 0.0;
			let mut weight = 0.0;
			for (i, &count) in read_histogram.iter().enumerate().skip(1) {
				let count = count as f32;
				let with_count = seen + count;
				if with_count >= range_end {
					let s = range_end - seen;
					weight += i as f32 * s;
					sum += s;
					break;
				}
				if with_count >= range_start {
					let s = (with_count - range_start).min(count);
					weight += i as f32 * s;
					sum += s;
				}
				seen = with_count;
			}

			if sum == 0.0 {
				*target_exposure = *exposure;
				return;
			}

			let exp_bin = weight / sum;
			let log = (exp_bin - 1.0) / 254.0;
			let target = log * (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE) + Self::MIN_EXPOSURE;
			let lum = Self::exposure_to_lum(2.0 * target + 10.0);
			let key = 1.03 - 2.0 / (2.0 + (lum + 1.0).log10());
			let comp = 6.0 * key - 2.5;
			*target_exposure = target - comp;

			let lerp = (1.0 - (-1.2 * dt).exp()).clamp(0.0, 1.0);
			*exposure = (1.0 - lerp) * *exposure + lerp * *target_exposure;
		});

		ExposureStats {
			exposure: ret_exp,
			target_exposure: target_exp,
			histogram: hist,
		}
	}

	pub unsafe fn destroy(self) { self.histogram.destroy(); }
}
