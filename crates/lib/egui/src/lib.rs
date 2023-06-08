#![feature(allocator_api)]

use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, cast_slice, NoUninit};
use egui::{
	epaint::{Primitive, Vertex},
	ClippedPrimitive,
	Color32,
	ImageData,
	Rect,
	TextureFilter,
	TextureId,
	TextureOptions,
	TexturesDelta,
};
use radiance_graph::{
	arena::{Arena, IteratorAlloc},
	device::{
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
		QueueType,
	},
	graph::{
		BufferUsage,
		BufferUsageType,
		ExecutionSnapshot,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Shader,
		UploadBufferDesc,
		VirtualResourceDesc,
		WriteId,
	},
	resource::{Image, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource, UploadBufferHandle},
	Result,
};
use radiance_shader_compiler::{
	c_str,
	runtime::{ShaderBlob, ShaderRuntime},
	shader,
};
use radiance_util::staging::{ImageStage, StageTicket, Staging};
use rustc_hash::FxHashMap;
use tracing::{span, Level};
use vek::Vec2;

const SHADERS: ShaderBlob = shader!("radiance-egui");

const VERTEX_BUFFER_START_CAPACITY: usize = (std::mem::size_of::<Vertex>() * 1024) as _;
const INDEX_BUFFER_START_CAPACITY: usize = (std::mem::size_of::<u32>() * 1024 * 3) as _;

pub struct ScreenDescriptor {
	pub physical_size: Vec2<u32>,
	pub scaling: f32,
}

pub struct Renderer {
	staging: Staging,
	snapshot: ExecutionSnapshot,
	images: FxHashMap<TextureId, (Image, Vec2<u32>, ImageView, SamplerId)>,
	samplers: FxHashMap<TextureOptions, (vk::Sampler, SamplerId)>,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	format: vk::Format,
	vertex_size: usize,
	index_size: usize,
}

struct PassIO {
	vertex: WriteId<UploadBufferHandle>,
	index: WriteId<UploadBufferHandle>,
	out: WriteId<ImageView>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstantsStatic {
	screen_size: Vec2<u32>,
	vertex_buffer: BufferId,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstantsDynamic {
	image: ImageId,
	sampler: SamplerId,
}

impl Renderer {
	pub fn new(device: &Device, output_format: vk::Format) -> Result<Self> {
		let (layout, pipeline) = unsafe {
			let rt = ShaderRuntime::new(device.device(), &[SHADERS]);

			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.descriptors().layout()])
					.push_constant_ranges(&[
						vk::PushConstantRange {
							stage_flags: vk::ShaderStageFlags::VERTEX,
							offset: 0,
							size: std::mem::size_of::<PushConstantsStatic>() as u32,
						},
						vk::PushConstantRange {
							stage_flags: vk::ShaderStageFlags::FRAGMENT,
							offset: std::mem::size_of::<PushConstantsStatic>() as u32,
							size: std::mem::size_of::<PushConstantsDynamic>() as u32,
						},
					]),
				None,
			)?;

			let pipeline = device
				.device()
				.create_graphics_pipelines(
					vk::PipelineCache::null(),
					&[vk::GraphicsPipelineCreateInfo::builder()
						.stages(&[
							rt.shader(c_str!("radiance-egui/vertex"), vk::ShaderStageFlags::VERTEX, None)
								.build(),
							rt.shader(c_str!("radiance-egui/pixel"), vk::ShaderStageFlags::FRAGMENT, None)
								.build(),
						])
						.vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::builder())
						.input_assembly_state(
							&vk::PipelineInputAssemblyStateCreateInfo::builder()
								.topology(vk::PrimitiveTopology::TRIANGLE_LIST),
						)
						.viewport_state(
							&vk::PipelineViewportStateCreateInfo::builder()
								.viewports(&[vk::Viewport::builder().build()])
								.scissors(&[vk::Rect2D::builder().build()]),
						)
						.rasterization_state(
							&vk::PipelineRasterizationStateCreateInfo::builder()
								.polygon_mode(vk::PolygonMode::FILL)
								.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
								.cull_mode(vk::CullModeFlags::NONE)
								.line_width(1.0),
						)
						.multisample_state(
							&vk::PipelineMultisampleStateCreateInfo::builder()
								.rasterization_samples(vk::SampleCountFlags::TYPE_1),
						)
						.color_blend_state(
							&vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
								vk::PipelineColorBlendAttachmentState::builder()
									.color_write_mask(
										vk::ColorComponentFlags::R
											| vk::ColorComponentFlags::G | vk::ColorComponentFlags::B
											| vk::ColorComponentFlags::A,
									)
									.blend_enable(true)
									.src_color_blend_factor(vk::BlendFactor::ONE)
									.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
									.color_blend_op(vk::BlendOp::ADD)
									.src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
									.dst_alpha_blend_factor(vk::BlendFactor::ONE)
									.alpha_blend_op(vk::BlendOp::ADD)
									.build(),
							]),
						)
						.dynamic_state(
							&vk::PipelineDynamicStateCreateInfo::builder()
								.dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
						)
						.layout(layout)
						.push_next(
							&mut vk::PipelineRenderingCreateInfo::builder().color_attachment_formats(&[output_format]),
						)
						.build()],
					None,
				)
				.unwrap()[0];

			rt.destroy(device.device());

			(layout, pipeline)
		};

		Ok(Self {
			staging: Staging::new(device)?,
			snapshot: ExecutionSnapshot::default(),
			images: FxHashMap::default(),
			samplers: FxHashMap::default(),
			layout,
			pipeline,
			format: output_format,
			vertex_size: VERTEX_BUFFER_START_CAPACITY,
			index_size: INDEX_BUFFER_START_CAPACITY,
		})
	}

	pub fn render<'pass, D: VirtualResourceDesc<Resource = ImageView>>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, device: &Device, tris: Vec<ClippedPrimitive>,
		delta: TexturesDelta, screen: ScreenDescriptor, out: D,
	) {
		let span = span!(Level::TRACE, "setup ui pass");
		let _e = span.enter();

		self.snapshot = frame.graph().snapshot();
		let (vertices, indices) = tris
			.iter()
			.filter_map(|x| match &x.primitive {
				Primitive::Mesh(m) => Some((m.vertices.len(), m.indices.len())),
				_ => None,
			})
			.fold((0, 0), |(v1, i1), (v2, i2)| (v1 + v2, i1 + i2));
		let vertex_size = vertices * std::mem::size_of::<Vertex>();
		if vertex_size > self.vertex_size {
			self.vertex_size *= 2;
		}
		let index_size = indices * std::mem::size_of::<u32>();
		if index_size > self.index_size {
			self.index_size *= 2;
		}

		let arena = frame.arena();
		let mut pass = frame.pass("ui");

		let (_, vertex) = pass.output(
			UploadBufferDesc { size: self.vertex_size },
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);
		let (_, index) = pass.output(
			UploadBufferDesc { size: self.index_size },
			BufferUsage {
				usages: &[BufferUsageType::IndexBuffer],
			},
		);
		let (_, out) = pass.output(
			out,
			ImageUsage {
				format: self.format,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);

		self.staging.poll(device).unwrap();
		if let Some(ticket) = self.generate_images(device, arena, delta) {
			pass.wait_on(ticket.as_info());
		}

		pass.build(move |ctx| unsafe { self.execute(ctx, PassIO { vertex, index, out }, &tris, &screen) });
	}

	/// # Safety
	/// Appropriate synchronization must be performed.
	pub unsafe fn destroy(self, device: &Device) {
		self.staging.destroy(device);
		for (_, (image, _, view, _)) in self.images {
			view.destroy(device);
			image.destroy(device);
		}
		for (_, (sampler, id)) in self.samplers {
			unsafe {
				device.device().destroy_sampler(sampler, None);
				device.descriptors().return_sampler(id);
			}
		}
		device.device().destroy_pipeline_layout(self.layout, None);
		device.device().destroy_pipeline(self.pipeline, None);
	}

	unsafe fn execute(
		&mut self, mut ctx: PassContext, io: PassIO, tris: &[ClippedPrimitive], screen: &ScreenDescriptor,
	) {
		let vertex = ctx.write(io.vertex);
		let index = ctx.write(io.index);
		let out = ctx.write(io.out);

		Self::generate_buffers(vertex, index, tris);

		ctx.device.device().cmd_begin_rendering(
			ctx.buf,
			&vk::RenderingInfo::builder()
				.render_area(
					vk::Rect2D::builder()
						.extent(
							vk::Extent2D::builder()
								.width(screen.physical_size.x)
								.height(screen.physical_size.y)
								.build(),
						)
						.build(),
				)
				.layer_count(1)
				.color_attachments(&[vk::RenderingAttachmentInfo::builder()
					.image_view(out.view)
					.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.clear_value(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(vk::AttachmentStoreOp::STORE)
					.build()]),
		);
		ctx.device.device().cmd_set_viewport(
			ctx.buf,
			0,
			&[vk::Viewport {
				x: 0.0,
				y: 0.0,
				width: screen.physical_size.x as f32,
				height: screen.physical_size.y as f32,
				min_depth: 0.0,
				max_depth: 1.0,
			}],
		);
		ctx.device
			.device()
			.cmd_bind_pipeline(ctx.buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
		ctx.device.device().cmd_bind_descriptor_sets(
			ctx.buf,
			vk::PipelineBindPoint::GRAPHICS,
			self.layout,
			0,
			&[ctx.device.descriptors().set()],
			&[],
		);
		ctx.device.device().cmd_push_constants(
			ctx.buf,
			self.layout,
			vk::ShaderStageFlags::VERTEX,
			0,
			bytes_of(&PushConstantsStatic {
				screen_size: screen.physical_size,
				vertex_buffer: vertex.id.unwrap(),
			}),
		);
		ctx.device
			.device()
			.cmd_bind_index_buffer(ctx.buf, index.buffer, 0, vk::IndexType::UINT32);

		let mut start_index = 0;
		for prim in tris {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let rect = ScissorRect::new(&prim.clip_rect, screen);
					if rect.width == 0 || rect.height == 0 {
						continue;
					}

					ctx.device.device().cmd_set_scissor(
						ctx.buf,
						0,
						&[vk::Rect2D {
							extent: vk::Extent2D {
								width: rect.width,
								height: rect.height,
							},
							offset: vk::Offset2D {
								x: rect.x as _,
								y: rect.y as _,
							},
						}],
					);
					let (_, _, image, sampler) = &self.images[&m.texture_id];
					ctx.device.device().cmd_push_constants(
						ctx.buf,
						self.layout,
						vk::ShaderStageFlags::FRAGMENT,
						std::mem::size_of::<PushConstantsStatic>() as u32,
						bytes_of(&PushConstantsDynamic {
							image: image.id.unwrap(),
							sampler: *sampler,
						}),
					);
					ctx.device
						.device()
						.cmd_draw_indexed(ctx.buf, m.indices.len() as u32, 1, start_index, 0, 0);
					start_index += m.indices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}

		ctx.device.device().cmd_end_rendering(ctx.buf);
	}

	unsafe fn generate_buffers(
		mut vertex: UploadBufferHandle, mut index: UploadBufferHandle, tris: &[ClippedPrimitive],
	) {
		let span = span!(Level::TRACE, "upload ui buffers");
		let _e = span.enter();

		let mut vertices_written = 0;
		let mut vertex_slice = vertex.data.as_mut();
		let mut index_slice = index.data.as_mut();

		for prim in tris.iter() {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let bytes: &[u8] = cast_slice(&m.vertices);
					vertex_slice.write_all(bytes).unwrap();

					for i in m.indices.iter() {
						index_slice.write_all(bytes_of(&(i + vertices_written))).unwrap();
					}

					vertices_written += m.vertices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}
	}

	fn generate_images(&mut self, device: &Device, arena: &Arena, delta: TexturesDelta) -> Option<StageTicket> {
		let span = span!(Level::TRACE, "upload ui images");
		let _e = span.enter();

		let ticket = if !delta.set.is_empty() {
			let ticket = self
				.staging
				.stage(
					device,
					self.snapshot.as_submit_info().into_iter().collect_in(arena),
					|stage| {
						for (id, data) in delta.set {
							let (image, size, ..) = self.images.entry(id).or_insert_with(|| {
								let size = Vec2::new(data.image.width() as u32, data.image.height() as _);
								let image = Image::create(
									device,
									ImageDesc {
										format: vk::Format::R8G8B8A8_SRGB,
										size: vk::Extent3D {
											width: size.x,
											height: size.y,
											depth: 1,
										},
										levels: 1,
										layers: 1,
										samples: vk::SampleCountFlags::TYPE_1,
										usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
										..Default::default()
									},
								)
								.unwrap();

								let view = ImageView::create(
									device,
									ImageViewDesc {
										image: image.handle(),
										view_type: vk::ImageViewType::TYPE_2D,
										format: vk::Format::R8G8B8A8_SRGB,
										usage: ImageViewUsage::Sampled,
										aspect: vk::ImageAspectFlags::COLOR,
									},
								)
								.unwrap();

								fn map_filter(filter: TextureFilter) -> vk::Filter {
									match filter {
										TextureFilter::Nearest => vk::Filter::NEAREST,
										TextureFilter::Linear => vk::Filter::LINEAR,
									}
								}

								let (_, id) = self.samplers.entry(data.options).or_insert_with(|| unsafe {
									let sampler = device
										.device()
										.create_sampler(
											&vk::SamplerCreateInfo::builder()
												.mag_filter(map_filter(data.options.magnification))
												.min_filter(map_filter(data.options.minification))
												.mipmap_mode(vk::SamplerMipmapMode::LINEAR),
											None,
										)
										.unwrap();

									let id = device.descriptors().get_sampler(device, sampler);

									(sampler, id)
								});

								(image, size, view, *id)
							});

							let pos = data.pos.unwrap_or([0, 0]);
							let vec: Vec<Color32, &Arena>;
							let image_subresource = vk::ImageSubresourceLayers {
								aspect_mask: vk::ImageAspectFlags::COLOR,
								mip_level: 0,
								base_array_layer: 0,
								layer_count: 1,
							};
							stage.stage_image(
								match &data.image {
									ImageData::Color(c) => cast_slice(&c.pixels),
									ImageData::Font(f) => {
										vec = f.srgba_pixels(None).collect_in(arena);
										cast_slice(&vec)
									},
								},
								image.handle(),
								ImageStage {
									buffer_row_length: 0,
									buffer_image_height: 0,
									image_subresource,
									image_offset: vk::Offset3D {
										x: pos[0] as i32,
										y: pos[1] as i32,
										z: 0,
									},
									image_extent: vk::Extent3D {
										width: data.image.width() as u32,
										height: data.image.height() as u32,
										depth: 1,
									},
								},
								pos == [0, 0]
									&& *size == Vec2::new(data.image.width() as u32, data.image.height() as _),
								QueueType::Graphics,
								&[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
								&[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
							)?;
						}

						Ok(())
					},
				)
				.unwrap();
			Some(ticket)
		} else {
			None
		};

		for free in delta.free {
			let (image, _, view, _) = self.images.remove(&free).unwrap();
			unsafe {
				image.destroy(device);
				view.destroy(device);
			}
		}

		ticket
	}
}

struct ScissorRect {
	x: u32,
	y: u32,
	width: u32,
	height: u32,
}

impl ScissorRect {
	fn new(clip_rect: &Rect, screen: &ScreenDescriptor) -> Self {
		let clip_min_x = screen.scaling * clip_rect.min.x;
		let clip_min_y = screen.scaling * clip_rect.min.y;
		let clip_max_x = screen.scaling * clip_rect.max.x;
		let clip_max_y = screen.scaling * clip_rect.max.y;

		let clip_min_x = clip_min_x.round() as u32;
		let clip_min_y = clip_min_y.round() as u32;
		let clip_max_x = clip_max_x.round() as u32;
		let clip_max_y = clip_max_y.round() as u32;

		let clip_min_x = clip_min_x.clamp(0, screen.physical_size.x);
		let clip_min_y = clip_min_y.clamp(0, screen.physical_size.y);
		let clip_max_x = clip_max_x.clamp(clip_min_x, screen.physical_size.x);
		let clip_max_y = clip_max_y.clamp(clip_min_y, screen.physical_size.y);

		Self {
			x: clip_min_x,
			y: clip_min_y,
			width: clip_max_x - clip_min_x,
			height: clip_max_y - clip_min_y,
		}
	}
}
