use ash::vk;
use bytemuck::NoUninit;
use rad_graph::{
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferUsage, Frame, ImageDesc, ImageUsage, PassContext, Res, Shader},
	resource::{BufferHandle, GpuPtr, ImageView},
	util::{
		pass::{Attachment, Load},
		render::FullscreenPass,
	},
	Result,
};

use crate::{
	mesh::{CameraData, GpuVisBufferReaderDebug, RenderOutput},
	scene::GpuInstance,
	util::SliceWriter,
};

#[derive(Copy, Clone)]
pub enum DebugVis {
	Triangles,
	Meshlets,
	Overdraw(f32),
	HwSw,
	Normals,
	Uvs,
	Error,
	BaseColor,
	Roughness,
	Metallic,
	Emissive,
}

impl DebugVis {
	pub fn requires_debug_info(self) -> bool { matches!(self, Self::Overdraw(..) | Self::HwSw) }

	pub fn to_u32(self) -> u32 {
		match self {
			DebugVis::Triangles => 0,
			DebugVis::Meshlets => 1,
			DebugVis::Overdraw(_) => 2,
			DebugVis::HwSw => 3,
			DebugVis::Normals => 4,
			DebugVis::Uvs => 5,
			DebugVis::Error => 6,
			DebugVis::BaseColor => 7,
			DebugVis::Roughness => 8,
			DebugVis::Metallic => 9,
			DebugVis::Emissive => 10,
		}
	}
}

pub struct DebugMesh {
	pass: FullscreenPass<PushConstants>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	camera: GpuPtr<CameraData>,
	read: GpuVisBufferReaderDebug,
	highlighted: GpuPtr<u32>,
	highlight_count: u32,
	ty: u32,
	overdraw_scale: f32,
	pad: u32,
}

impl DebugMesh {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "passes.debug.main",
					spec: &["passes.mesh.debug"],
				},
				&[vk::Format::R8G8B8A8_SRGB],
			)?,
		})
	}

	/// `highlights` must be sorted.
	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, vis: DebugVis, output: RenderOutput,
		highlights: impl ExactSizeIterator<Item = u32> + 'pass,
	) -> Res<ImageView> {
		let mut pass = frame.pass("debug mesh");

		pass.reference(output.scene.instances, BufferUsage::read(Shader::Fragment));
		pass.reference(output.camera, BufferUsage::read(Shader::Fragment));
		output.reader.add(&mut pass, Shader::Fragment, true);

		let desc = pass.desc(output.reader.visbuffer);
		let out = pass.resource(
			ImageDesc {
				format: vk::Format::R8G8B8A8_SRGB,
				..desc
			},
			ImageUsage::color_attachment(),
		);

		let highlight_buf = (highlights.len() > 0).then(|| {
			pass.resource(
				BufferDesc::upload((std::mem::size_of::<u32>() * highlights.len()) as _),
				BufferUsage::read(Shader::Fragment),
			)
		});

		pass.build(move |ctx| self.execute(ctx, vis, output, highlight_buf, highlights, out));
		out
	}

	fn execute<'pass>(
		&'pass self, mut pass: PassContext, vis: DebugVis, output: RenderOutput,
		highlight_buf: Option<Res<BufferHandle>>, highlights: impl Iterator<Item = u32> + 'pass, out: Res<ImageView>,
	) {
		unsafe {
			let highlight = highlight_buf.map(|x| pass.get(x));
			let mut count = 0;
			if let Some(mut h) = highlight {
				let mut w = SliceWriter::new(h.data.as_mut());
				for i in highlights {
					w.write(i);
					count += 1;
				}
			}

			let overdraw_scale = match vis {
				DebugVis::Overdraw(s) => s,
				_ => 0.0,
			};
			let instances = pass.get(output.scene.instances).ptr();
			let camera = pass.get(output.camera).ptr();
			let read = output.reader.get_debug(&mut pass);
			self.pass.run(
				&mut pass,
				&PushConstants {
					instances,
					camera,
					read,
					highlighted: highlight.map(|x| x.ptr()).unwrap_or(GpuPtr::null()),
					highlight_count: count,
					ty: vis.to_u32(),
					overdraw_scale,
					pad: 0,
				},
				&[Attachment {
					image: out,
					load: Load::Clear(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					}),
					store: true,
				}],
			);
		}
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
