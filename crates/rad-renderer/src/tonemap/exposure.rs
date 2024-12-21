use ash::vk;
use bytemuck::{from_bytes, NoUninit};
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
	avg_lum: GpuPtr<f32>,
	input: ImageId,
	min_log_lum: f32,
	inv_log_lum_range: f32,
	lerp_coeff: f32,
}

pub struct ExposureCalc {
	histogram: ComputePass<PushConstants>,
	average: ComputePass<PushConstants>,
	read_histogram: [u32; 256],
	exposure: f32,
}

pub struct ExposureStats {
	pub luminance: f32,
	pub histogram: [u32; 256],
}

impl ExposureCalc {
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
		let avg_lum_size = std::mem::size_of::<f32>() as u64;

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
		let avg_lum = pass.resource(
			BufferDesc {
				size: avg_lum_size,
				loc: BufferLoc::GpuOnly,
				persist: Some("average luminance"),
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		// TODO: stop using direct commands here and in mesh/setup.rs
		pass.build(move |mut pass| unsafe {
			let buf = pass.get(histogram);
			pass.device
				.device()
				.cmd_fill_buffer(pass.buf, buf.buffer, 0, buf.size(), 0);
			if pass.is_uninit(avg_lum) {
				let buf = pass.get(avg_lum);
				pass.device
					.device()
					.cmd_fill_buffer(pass.buf, buf.buffer, 0, buf.size(), 0);
			}
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
		let min_log_lum = -8.0;
		let max_log_lum = 3.5;
		let inv_log_lum_range = 1.0 / (max_log_lum - min_log_lum);
		pass.build(move |mut pass| {
			let input = pass.get(input).id.unwrap();
			let histogram = pass.get(histogram).ptr();
			hist.dispatch(
				&mut pass,
				&PushConstants {
					histogram,
					avg_lum: GpuPtr::null(),
					input,
					lerp_coeff: 0.0,
					min_log_lum,
					inv_log_lum_range,
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
			avg_lum,
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
			let avg_lum = pass.get(avg_lum).ptr();
			average.dispatch(
				&mut pass,
				&PushConstants {
					histogram,
					avg_lum,
					input,
					min_log_lum,
					inv_log_lum_range,
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
		let avg_lum_read = pass.resource(
			BufferDesc {
				size: avg_lum_size,
				loc: BufferLoc::Readback,
				persist: Some("exposure readback"),
			},
			usage,
		);
		let usage = BufferUsage {
			usages: &[BufferUsageType::TransferRead],
		};
		pass.reference(histogram, usage);
		pass.reference(avg_lum, usage);

		let exp = *exposure;
		let hist = *read_histogram;
		pass.build(move |mut pass| unsafe {
			let avg_lum = pass.get(avg_lum);
			let histogram = pass.get(histogram);

			let uninit = pass.is_uninit(histogram_read);
			let read = pass.get(histogram_read);
			pass.device.device().cmd_copy_buffer(
				pass.buf,
				histogram.buffer,
				read.buffer,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: histogram_size,
				}],
			);
			if !uninit {
				*read_histogram = *from_bytes(&read.data.as_ref()[..histogram_size as usize]);
			}

			let uninit = pass.is_uninit(avg_lum_read);
			let read = pass.get(avg_lum_read);
			pass.device.device().cmd_copy_buffer(
				pass.buf,
				avg_lum.buffer,
				read.buffer,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: avg_lum_size,
				}],
			);
			if !uninit {
				*exposure = *from_bytes(&read.data.as_ref()[..avg_lum_size as usize]);
			}
		});

		(
			avg_lum,
			ExposureStats {
				luminance: exp,
				histogram: hist,
			},
		)
	}
}
