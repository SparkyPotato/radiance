use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{BufferDesc, BufferUsage, Frame, ImageUsage, PassContext, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView, ImageViewDescUnnamed, ImageViewUsage, Subresource},
	util::compute::ComputePass,
	Result,
};
use vek::Vec2;

pub struct HzbGen {
	pass: ComputePass<PushConstants>,
	hzb_sample: SamplerId,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	atomic: GpuPtr<u32>,
	visbuffer: StorageImageId,
	outs: [Option<StorageImageId>; 12],
	mips: u32,
	target: u32,
	_pad: u32,
}

struct PassIO {
	atomic: Res<BufferHandle>,
	visbuffer: Res<ImageView>,
	out: Res<ImageView>,
	size: Vec2<u32>,
	levels: u32,
}

impl HzbGen {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: ComputePass::new(
				device,
				ShaderInfo {
					shader: "passes.mesh.hzb.main",
					spec: &[],
				},
			)?,
			hzb_sample: device.sampler(SamplerDesc {
				mag_filter: vk::Filter::LINEAR,
				min_filter: vk::Filter::LINEAR,
				mipmap_mode: vk::SamplerMipmapMode::NEAREST,
				address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
				reduction_mode: vk::SamplerReductionMode::MIN,
				..Default::default()
			}),
		})
	}

	pub fn sampler(&self) -> SamplerId { self.hzb_sample }

	pub fn run<'pass>(&'pass self, frame: &mut Frame<'pass, '_>, visbuffer: Res<ImageView>, out: Res<ImageView>) {
		frame.start_region("generate hzb");

		let atomic = frame.stage_buffer_new(
			"zero atomic",
			BufferDesc::gpu(std::mem::size_of::<u32>() as _),
			0,
			&[0, 0, 0, 0],
		);

		let mut pass = frame.pass("run");
		pass.reference(visbuffer, ImageUsage::read_2d(Shader::Compute));
		pass.reference(out, ImageUsage::read_write_2d(Shader::Compute));
		pass.reference(atomic, BufferUsage::read_write(Shader::Compute));

		let desc = pass.desc(out);
		let size = Vec2::new(desc.size.width, desc.size.height) * 2;
		pass.build(move |ctx| {
			self.execute(
				ctx,
				PassIO {
					atomic,
					visbuffer,
					out,
					size,
					levels: desc.levels,
				},
			)
		});

		frame.end_region();
	}

	fn execute(&self, mut pass: PassContext, io: PassIO) {
		let atomic = pass.get(io.atomic);
		let visbuffer = pass.get(io.visbuffer);
		let out = pass.get(io.out);

		let mut outs = [None; 12];
		let mut s = io.size;
		for i in 0..io.levels {
			if s.x > 1 {
				s.x /= 2;
			}
			if s.y > 1 {
				s.y /= 2;
			}
			let dev = pass.device;
			outs[i as usize] = Some(
				pass.caches()
					.image_views
					.get(
						dev,
						ImageViewDescUnnamed {
							image: out.image,
							view_type: vk::ImageViewType::TYPE_2D,
							format: vk::Format::R32_SFLOAT,
							usage: ImageViewUsage::Storage,
							size: vk::Extent3D::default().width(s.x).height(s.y).depth(1),
							subresource: Subresource {
								first_mip: i,
								mip_count: 1,
								..Default::default()
							},
						},
					)
					.unwrap()
					.0
					.storage_id
					.unwrap(),
			);
		}

		let x = io.size.x.div_ceil(64);
		let y = io.size.y.div_ceil(64);
		let push = PushConstants {
			atomic: atomic.ptr(),
			visbuffer: visbuffer.storage_id.unwrap(),
			outs,
			mips: io.levels,
			target: x * y - 1,
			_pad: 0,
		};
		self.pass.dispatch(&mut pass, &push, x, y, 1);
	}

	pub fn destroy(self) {
		unsafe {
			self.pass.destroy();
		}
	}
}
