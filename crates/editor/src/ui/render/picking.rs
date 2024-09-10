use ash::vk;
use bytemuck::{bytes_of, from_bytes, NoUninit};
use radiance_graph::{
	device::{Device, Pipeline, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, Frame},
	resource::GpuPtr,
	sync::Shader,
	Result,
};
use radiance_passes::mesh::{GpuVisBufferReader, VisBufferReader};
use vek::Vec2;

pub struct Picker {
	frames: u64,
	selection: Option<u32>,
	layout: vk::PipelineLayout,
	pipeline: Pipeline,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	reader: GpuVisBufferReader,
	pix: Vec2<u32>,
	should_pick: u32,
	_pad: u32,
	ret: GpuPtr<u32>,
}

impl Picker {
	pub fn new(device: &Device) -> Result<Self> {
		let layout = unsafe {
			device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[vk::PushConstantRange::default()
						.stage_flags(vk::ShaderStageFlags::COMPUTE)
						.size(std::mem::size_of::<PushConstants>() as u32)]),
				None,
			)?
		};
		Ok(Self {
			frames: 0,
			selection: None,
			layout,
			pipeline: device.compute_pipeline(
				layout,
				ShaderInfo {
					shader: "editor.mousepick.main",
					spec: &[],
				},
			)?,
		})
	}

	pub fn get_sel(&self) -> Option<u32> { self.selection }

	pub fn select(&mut self, index: u32) { self.selection = Some(index); }

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, visbuffer: VisBufferReader, click: Option<egui::Vec2>,
	) -> Option<u32> {
		let mut pass = frame.pass("mousepick");
		let ret = pass.resource(
			BufferDesc {
				size: std::mem::size_of::<u32>() as _,
				loc: BufferLoc::Readback,
				persist: Some("mousepick readback"),
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageWrite(Shader::Compute)],
			},
		);
		visbuffer.add(&mut pass, Shader::Compute, false);

		let sel = self.selection;
		pass.build(move |mut pass| unsafe {
			let dev = pass.device.device();
			let buf = pass.buf;

			let ret = pass.get(ret);
			let prev: u32 = *from_bytes(&ret.data.as_ref()[..4]);
			if prev != u32::MAX && self.frames > 2 {
				self.selection = (prev != u32::MAX - 1).then_some(prev);
			}

			let pix = click.map(|x| Vec2::new(x.x as _, x.y as _)).unwrap_or_default();
			dev.cmd_bind_descriptor_sets(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[pass.device.descriptors().set()],
				&[],
			);
			dev.cmd_bind_pipeline(buf, vk::PipelineBindPoint::COMPUTE, self.pipeline.get());
			dev.cmd_push_constants(
				buf,
				self.layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				bytes_of(&PushConstants {
					reader: visbuffer.get(&mut pass),
					pix,
					should_pick: click.is_some() as _,
					_pad: 0,
					ret: ret.ptr(),
				}),
			);
			dev.cmd_dispatch(buf, 1, 1, 1);
			self.frames += 1;
		});
		sel
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.pipeline.destroy();
		device.device().destroy_pipeline_layout(self.layout, None);
	}
}
