use bytemuck::NoUninit;
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferDesc, BufferUsage, Frame, ImageUsage, Persist, Res},
	resource::{BufferHandle, GpuPtr, ImageView},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct HPushConstants {
	histogram: GpuPtr<u32>,
	input: ImageId,
	min_exp: f32,
	inv_exp_range: f32,
	_pad: u32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct EPushConstants {
	histogram: GpuPtr<u32>,
	exposure: GpuPtr<f32>,
	histogram_min: f32,
	histogram_max: f32,
	compensation: f32,
	min_exp: f32,
	exp_range: f32,
	lerp_coeff: f32,
}

pub struct ExposureCalc {
	histogram: ComputePass<HPushConstants>,
	exposure: ComputePass<EPushConstants>,
	exposure_value: Persist<BufferHandle>,
	histogram_readback: Persist<BufferHandle>,
	exposure_readback: Persist<BufferHandle>,
	curr_exposure: f32,
	target_exposure: f32,
	scene_exposure: f32,
	read_histogram: [u32; 256],
}

pub struct ExposureStats {
	pub exposure: f32,
	pub target_exposure: f32,
	pub scene_exposure: f32,
	pub histogram: [u32; 256],
}

impl ExposureCalc {
	pub const MAX_EXPOSURE: f32 = 18.0;
	pub const MAX_HISTOGRAM_RANGE: f32 = 0.95;
	pub const MIN_EXPOSURE: f32 = -6.0;
	pub const MIN_HISTOGRAM_RANGE: f32 = 0.6;

	pub fn bin_to_exposure(bin: f32) -> f32 {
		let log = (bin - 1.0) / 254.0;
		log.mul_add(Self::MAX_EXPOSURE - Self::MIN_EXPOSURE, Self::MIN_EXPOSURE)
	}

	pub fn exposure_to_bin(exp: f32) -> f32 {
		((exp - Self::MIN_EXPOSURE) / (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE)).mul_add(254.0, 1.0)
	}

	pub fn exposure_to_lum(exp: f32) -> f32 { (exp - 3.0).exp2() }

	pub fn exposure_compensation(exp: f32) -> f32 {
		let lum = Self::exposure_to_lum(exp.mul_add(2.0, 10.0));
		let key = 1.03 - 2.0 / (2.0 + (lum + 1.0).log10());
		key.mul_add(6.0, -2.5)
	}

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			histogram: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.exposure.histogram",
					spec: &[],
				},
			)?,
			exposure: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.tonemap.exposure.exposure",
					spec: &[],
				},
			)?,
			exposure_value: Persist::new(),
			histogram_readback: Persist::new(),
			exposure_readback: Persist::new(),
			curr_exposure: 0.0,
			target_exposure: 0.0,
			scene_exposure: 0.0,
			read_histogram: [0; 256],
		})
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, input: Res<ImageView>, ec: f32, dt: f32,
	) -> (Res<BufferHandle>, ExposureStats) {
		frame.start_region("exposure");

		let Self {
			histogram: hist,
			exposure: exp,
			exposure_value,
			histogram_readback,
			exposure_readback,
			curr_exposure,
			target_exposure,
			scene_exposure,
			read_histogram,
		} = self;

		let histogram_size = std::mem::size_of::<u32>() as u64 * 256;

		let mut pass = frame.pass("zero data");
		let histogram = pass.resource(BufferDesc::gpu(histogram_size), BufferUsage::transfer_write());
		let exposure = pass.resource(
			BufferDesc::gpu(std::mem::size_of::<f32>() as u64 * 3).persist(*exposure_value),
			BufferUsage::transfer_write(),
		);
		pass.build(move |mut pass| {
			pass.zero(histogram);
			if pass.is_uninit(exposure) {
				pass.zero(exposure);
			}
		});

		let mut pass = frame.pass("generate histogram");
		pass.reference(input, ImageUsage::sampled_2d(Shader::Compute));
		pass.reference(histogram, BufferUsage::read_write(Shader::Compute));
		let size = pass.desc(input).size;
		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let histogram = pass.get(histogram).ptr();
			hist.dispatch(
				&mut pass,
				&HPushConstants {
					histogram,
					input,
					min_exp: Self::MIN_EXPOSURE,
					inv_exp_range: 1.0 / (Self::MAX_EXPOSURE - Self::MIN_EXPOSURE),
					_pad: 0,
				},
				size.width.div_ceil(16),
				size.height.div_ceil(16),
				1,
			)
		});

		let mut pass = frame.pass("calc exposure");
		pass.reference(histogram, BufferUsage::read(Shader::Compute));
		pass.reference(exposure, BufferUsage::read_write(Shader::Compute));
		pass.build(move |mut pass| {
			let histogram = pass.get(histogram).ptr();
			let exposure = pass.get(exposure).ptr();
			exp.dispatch(
				&mut pass,
				&EPushConstants {
					histogram,
					exposure,
					histogram_min: Self::MIN_HISTOGRAM_RANGE,
					histogram_max: Self::MAX_HISTOGRAM_RANGE,
					compensation: ec,
					min_exp: Self::MIN_EXPOSURE,
					exp_range: Self::MAX_EXPOSURE - Self::MIN_EXPOSURE,
					lerp_coeff: (1.0 - (-1.2 * dt).exp()).clamp(0.0, 1.0),
				},
				1,
				1,
				1,
			);
		});

		let mut pass = frame.pass("readback exposure");
		pass.reference(histogram, BufferUsage::transfer_read());
		pass.reference(exposure, BufferUsage::transfer_read());
		let histogram_read = pass.resource(
			BufferDesc::readback(histogram_size, *histogram_readback),
			BufferUsage::transfer_write(),
		);
		let exposure_read = pass.resource(
			BufferDesc::readback(std::mem::size_of::<f32>() as u64 * 3, *exposure_readback),
			BufferUsage::transfer_write(),
		);

		let ret_exp = *curr_exposure;
		let target_exp = *target_exposure;
		let scene_exp = *scene_exposure;
		let hist = *read_histogram;
		pass.build(move |mut pass| {
			pass.copy_full_buffer(histogram, histogram_read, 0);
			*read_histogram = pass.readback(histogram_read, 0);
			pass.copy_full_buffer(exposure, exposure_read, 0);
			[*curr_exposure, *target_exposure, *scene_exposure] = pass.readback(exposure_read, 0);
		});

		frame.end_region();

		(
			exposure,
			ExposureStats {
				exposure: ret_exp,
				target_exposure: target_exp,
				scene_exposure: scene_exp,
				histogram: hist,
			},
		)
	}

	pub unsafe fn destroy(self) { unsafe { self.histogram.destroy(); }}
}
