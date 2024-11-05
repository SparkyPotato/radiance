use ash::vk;
use bytemuck::NoUninit;
use radiance_asset::{io::SliceWriter, scene::GpuInstance};
use radiance_graph::{
	device::{Device, GraphicsPipelineDesc, ShaderInfo},
	graph::{
		BufferDesc,
		BufferLoc,
		BufferUsage,
		BufferUsageType,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Res,
		Shader,
	},
	resource::{BufferHandle, GpuPtr, ImageView, Subresource},
	util::{
		pipeline::{no_blend, no_cull, simple_blend},
		render::RenderPass,
	},
	Result,
};

use crate::mesh::{CameraData, GpuVisBufferReaderDebug, RenderOutput};

#[derive(Copy, Clone)]
pub enum DebugVis {
	Triangles,
	Meshlets,
	Overdraw(f32),
	HwSw,
	Normals,
	Uvs,
	Error,
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
		}
	}
}

pub struct DebugMesh {
	pass: RenderPass<PushConstants>,
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
			pass: RenderPass::new(
				device,
				GraphicsPipelineDesc {
					shaders: &[
						ShaderInfo {
							shader: "graph.util.screen",
							..Default::default()
						},
						ShaderInfo {
							shader: "passes.debug.main",
							spec: &["passes.mesh.debug"],
						},
					],
					raster: no_cull(),
					blend: simple_blend(&[no_blend()]),
					color_attachments: &[vk::Format::R8G8B8A8_SRGB],
					..Default::default()
				},
				false,
			)?,
		})
	}

	/// `highlights` must be sorted.
	pub fn run<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, vis: DebugVis, output: RenderOutput,
		highlights: impl ExactSizeIterator<Item = u32> + 'pass,
	) -> Res<ImageView> {
		let mut pass = frame.pass("debug mesh");

		let usage = BufferUsage {
			usages: &[BufferUsageType::ShaderStorageRead(Shader::Fragment)],
		};
		pass.reference(output.scene.instances, usage);
		pass.reference(output.camera, usage);
		output.reader.add(&mut pass, Shader::Fragment, true);

		let desc = pass.desc(output.reader.visbuffer);
		let out = pass.resource(
			ImageDesc {
				format: vk::Format::R8G8B8A8_SRGB,
				..desc
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		let highlight_buf = (highlights.len() > 0).then(|| {
			pass.resource(
				BufferDesc {
					size: (std::mem::size_of::<u32>() * highlights.len()) as u64,
					loc: BufferLoc::Upload,
					persist: None,
				},
				BufferUsage {
					usages: &[BufferUsageType::ShaderStorageRead(Shader::Fragment)],
				},
			)
		});

		pass.build(move |ctx| self.execute(ctx, vis, output, highlight_buf, highlights, out));
		out
	}

	fn execute<'pass>(
		&'pass self, mut pass: PassContext, vis: DebugVis, output: RenderOutput,
		highlight_buf: Option<Res<BufferHandle>>, highlights: impl Iterator<Item = u32> + 'pass, out: Res<ImageView>,
	) {
		let out = pass.get(out);

		unsafe {
			let highlight = highlight_buf.map(|x| pass.get(x));
			let mut count = 0;
			if let Some(mut h) = highlight {
				let mut w = SliceWriter::new(h.data.as_mut());
				for i in highlights {
					w.write(i).unwrap();
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
				&pass,
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
				vk::Extent2D {
					width: out.size.width,
					height: out.size.height,
				},
				&[vk::RenderingAttachmentInfo::default()
					.image_view(out.view)
					.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.clear_value(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(vk::AttachmentStoreOp::STORE)],
				|dev, buf| dev.device().cmd_draw(buf, 3, 1, 0, 0),
			);
		}
	}

	pub unsafe fn destroy(self) { self.pass.destroy(); }
}
