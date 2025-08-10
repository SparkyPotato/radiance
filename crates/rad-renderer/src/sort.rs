use bytemuck::NoUninit;
use rad_graph::{
	Result,
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferUsage, Frame, Res},
	resource::{BufferHandle, GpuPtr},
	sync::Shader,
	util::compute::ComputePass,
};

pub struct GpuSorter {
	histogram: ComputePass<PushConstants>,
	sort: ComputePass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	in_keys: GpuPtr<u64>,
	out_keys: GpuPtr<u64>,
	in_vals: GpuPtr<u32>,
	out_vals: GpuPtr<u32>,
	histogram: GpuPtr<u32>,
	elem_count: u32,
	shift: u32,
	workgroup_count: u32,
	num_blocks_per_workgroup: u32,
}

impl GpuSorter {
	const NUM_BLOCKS_PER_WG: u32 = 32;

	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			histogram: ComputePass::with_wave_32(
				device,
				ShaderInfo {
					shader: "sort.histogram",
					spec: &[],
				},
			)?,
			sort: ComputePass::with_wave_32(
				device,
				ShaderInfo {
					shader: "sort.sort",
					spec: &[],
				},
			)?,
		})
	}

	pub fn sort<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, mut keys: Res<BufferHandle>, mut vals: Res<BufferHandle>,
		elem_count: u32,
	) -> (Res<BufferHandle>, Res<BufferHandle>) {
		frame.start_region("gpu sort");

		let workgroup_count = elem_count.div_ceil(Self::NUM_BLOCKS_PER_WG).div_ceil(256);

		for i in 0..8 {
			let mut push = PushConstants {
				in_keys: GpuPtr::null(),
				out_keys: GpuPtr::null(),
				in_vals: GpuPtr::null(),
				out_vals: GpuPtr::null(),
				histogram: GpuPtr::null(),
				elem_count,
				shift: i * 8,
				workgroup_count,
				num_blocks_per_workgroup: Self::NUM_BLOCKS_PER_WG,
			};

			let mut pass = frame.pass("histogram");
			pass.reference(keys, BufferUsage::read(Shader::Compute));
			let histogram = pass.resource(
				BufferDesc::gpu(workgroup_count as u64 * 256 * std::mem::size_of::<u32>() as u64),
				BufferUsage::write(Shader::Compute),
			);
			pass.build(move |mut pass| {
				push.in_keys = pass.get(keys).ptr();
				push.histogram = pass.get(histogram).ptr();
				self.histogram.dispatch(&mut pass, &push, workgroup_count, 1, 1);
			});

			let mut pass = frame.pass("sort");
			pass.reference(keys, BufferUsage::read(Shader::Compute));
			pass.reference(vals, BufferUsage::read(Shader::Compute));
			pass.reference(histogram, BufferUsage::read(Shader::Compute));
			let out_keys = pass.resource(
				BufferDesc::gpu(elem_count as u64 * std::mem::size_of::<u64>() as u64),
				BufferUsage::write(Shader::Compute),
			);
			let out_vals = pass.resource(
				BufferDesc::gpu(elem_count as u64 * std::mem::size_of::<u32>() as u64),
				BufferUsage::write(Shader::Compute),
			);
			pass.build(move |mut pass| {
				push.in_keys = pass.get(keys).ptr();
				push.out_keys = pass.get(out_keys).ptr();
				push.in_vals = pass.get(vals).ptr();
				push.out_vals = pass.get(out_vals).ptr();
				push.histogram = pass.get(histogram).ptr();
				self.sort.dispatch(&mut pass, &push, workgroup_count, 1, 1);
			});
			keys = out_keys;
			vals = out_vals;
		}
		frame.end_region();

		(keys, vals)
	}
}
