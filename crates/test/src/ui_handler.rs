use std::io::Write;

use ash::vk::{ImageAspectFlags, ImageViewType};
use bytemuck::{bytes_of, cast_slice};
use egui::{
	epaint::{Primitive, Vertex},
	ClippedPrimitive,
	Context,
	TexturesDelta,
};
use radiance_graph::{
	graph::{
		BufferUsage,
		BufferUsageType,
		ExternalImage,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Shader,
		UploadBufferDesc,
		WriteId,
	},
	resource::{ImageView, UploadBufferHandle},
};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::window::Window;

pub struct UiHandler {
	ctx: Context,
	platform_state: egui_winit::State,
}

struct PassIO {
	vertex: WriteId<UploadBufferHandle>,
	index: WriteId<UploadBufferHandle>,
	out: WriteId<ImageView>,
}

impl UiHandler {
	pub fn new(event_loop: &EventLoop<()>) -> Self {
		Self {
			ctx: Context::default(),
			platform_state: egui_winit::State::new(event_loop),
		}
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, out: ExternalImage, window: &Window, run: impl FnOnce(&Context),
	) {
		tracy::zone!("UI Render");

		let output = self.ctx.run(self.platform_state.take_egui_input(&window.window), run);
		self.platform_state
			.handle_platform_output(&window.window, &self.ctx, output.platform_output);

		let tris = {
			tracy::zone!("Tessellate shapes");
			self.ctx.tessellate(output.shapes)
		};

		let (vertices, indices) = tris
			.iter()
			.filter_map(|x| match &x.primitive {
				Primitive::Mesh(m) => Some((m.vertices.len(), m.indices.len())),
				_ => None,
			})
			.fold((0, 0), |(v1, i1), (v2, i2)| (v1 + v2, i1 + i2));
		let vertex_size = vertices * std::mem::size_of::<Vertex>();
		let index_size = indices * std::mem::size_of::<u32>();

		let mut pass = frame.pass("UI Render");
		let (_, vertex) = pass.output(
			UploadBufferDesc { size: vertex_size },
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);
		let (_, index) = pass.output(
			UploadBufferDesc { size: index_size },
			BufferUsage {
				usages: &[BufferUsageType::IndexBuffer],
			},
		);
		let (_, out) = pass.output(
			out,
			ImageUsage {
				format: window.format(),
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: ImageViewType::TYPE_2D,
				aspect: ImageAspectFlags::COLOR,
			},
		);
		pass.build(|ctx| unsafe { self.render(ctx, PassIO { vertex, index, out }, output.textures_delta, tris) });
	}

	unsafe fn render(
		&mut self, mut ctx: PassContext, io: PassIO, textures_delta: TexturesDelta, tris: Vec<ClippedPrimitive>,
	) {
		let vertex = ctx.write(io.vertex);
		let index = ctx.write(io.index);
		let out = ctx.write(io.out);

		Self::generate_buffers(vertex, index, &tris);
	}

	unsafe fn generate_buffers(
		mut vertex: UploadBufferHandle, mut index: UploadBufferHandle, tris: &[ClippedPrimitive],
	) {
		let mut vertices_written = 0;
		let mut vertex_slice = vertex.data.as_mut();
		let mut index_slice = index.data.as_mut();

		for prim in tris.iter() {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let bytes: &[u8] = cast_slice(&m.vertices);
					vertex_slice.write(bytes).unwrap();

					for i in m.indices.iter() {
						index_slice.write(bytes_of(&(i + vertices_written))).unwrap();
					}

					vertices_written += m.vertices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}
	}

	pub fn on_event(&mut self, event: &WindowEvent) { let _ = self.platform_state.on_event(&self.ctx, event); }
}
