#![feature(allocator_api)]

use std::io::Write;

use ash::vk::{
	AccessFlags2,
	AttachmentLoadOp,
	AttachmentStoreOp,
	BlendFactor,
	BlendOp,
	ClearColorValue,
	ClearValue,
	ColorComponentFlags,
	CommandBufferBeginInfo,
	CommandBufferSubmitInfo,
	CommandBufferUsageFlags,
	CullModeFlags,
	DependencyInfo,
	DynamicState,
	Extent2D,
	Extent3D,
	Fence,
	Filter,
	Format,
	FrontFace,
	GraphicsPipelineCreateInfo,
	ImageAspectFlags,
	ImageLayout,
	ImageMemoryBarrier2,
	ImageSubresourceLayers,
	ImageSubresourceRange,
	ImageUsageFlags,
	ImageViewType,
	IndexType,
	Offset2D,
	Offset3D,
	Pipeline,
	PipelineBindPoint,
	PipelineCache,
	PipelineColorBlendAttachmentState,
	PipelineColorBlendStateCreateInfo,
	PipelineDynamicStateCreateInfo,
	PipelineInputAssemblyStateCreateInfo,
	PipelineLayout,
	PipelineLayoutCreateInfo,
	PipelineMultisampleStateCreateInfo,
	PipelineRasterizationStateCreateInfo,
	PipelineRenderingCreateInfo,
	PipelineStageFlags2,
	PipelineVertexInputStateCreateInfo,
	PipelineViewportStateCreateInfo,
	PolygonMode,
	PrimitiveTopology,
	PushConstantRange,
	Rect2D,
	RenderingAttachmentInfo,
	RenderingInfo,
	SampleCountFlags,
	SamplerMipmapMode,
	SemaphoreSubmitInfo,
	ShaderStageFlags,
	SubmitInfo2,
	Viewport,
};
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
		cmd::CommandPool,
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
	},
	graph::{
		BufferUsage,
		BufferUsageType,
		ExecutionSnapshot,
		ExternalImage,
		ExternalSync,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Shader,
		UploadBufferDesc,
		VirtualResourceDesc,
		WriteId,
	},
	resource::{
		Image,
		ImageDesc,
		ImageView,
		ImageViewDesc,
		ImageViewUsage,
		Resource,
		Sampler,
		SamplerDesc,
		UploadBufferHandle,
	},
	Result,
};
use radiance_shader_compiler::{
	c_str,
	runtime::{ShaderBlob, ShaderRuntime},
	shader,
};
use radiance_util::staging::{ImageStage, Staging};
use rustc_hash::FxHashMap;
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
	pool: CommandPool,
	snapshot: ExecutionSnapshot,
	images: FxHashMap<TextureId, (Image, ImageView, SamplerId)>,
	samplers: FxHashMap<TextureOptions, Sampler>,
	layout: PipelineLayout,
	pipeline: Pipeline,
	format: Format,
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
	pub fn new(device: &Device, output_format: Format) -> Result<Self> {
		let (layout, pipeline) = unsafe {
			let rt = ShaderRuntime::new(device.device(), &[SHADERS]);

			let layout = device.device().create_pipeline_layout(
				&PipelineLayoutCreateInfo::builder()
					.set_layouts(&[device.base_descriptors().layout()])
					.push_constant_ranges(&[
						PushConstantRange {
							stage_flags: ShaderStageFlags::VERTEX,
							offset: 0,
							size: std::mem::size_of::<PushConstantsStatic>() as u32,
						},
						PushConstantRange {
							stage_flags: ShaderStageFlags::FRAGMENT,
							offset: std::mem::size_of::<PushConstantsStatic>() as u32,
							size: std::mem::size_of::<PushConstantsDynamic>() as u32,
						},
					]),
				None,
			)?;

			let pipeline = device
				.device()
				.create_graphics_pipelines(
					PipelineCache::null(),
					&[GraphicsPipelineCreateInfo::builder()
						.stages(&[
							rt.shader(c_str!("radiance-egui/vertex"), ShaderStageFlags::VERTEX, None)
								.build(),
							rt.shader(c_str!("radiance-egui/pixel"), ShaderStageFlags::FRAGMENT, None)
								.build(),
						])
						.vertex_input_state(&PipelineVertexInputStateCreateInfo::builder())
						.input_assembly_state(
							&PipelineInputAssemblyStateCreateInfo::builder().topology(PrimitiveTopology::TRIANGLE_LIST),
						)
						.viewport_state(
							&PipelineViewportStateCreateInfo::builder()
								.viewports(&[Viewport::builder().build()])
								.scissors(&[Rect2D::builder().build()]),
						)
						.rasterization_state(
							&PipelineRasterizationStateCreateInfo::builder()
								.polygon_mode(PolygonMode::FILL)
								.front_face(FrontFace::COUNTER_CLOCKWISE)
								.cull_mode(CullModeFlags::NONE)
								.line_width(1.0),
						)
						.multisample_state(
							&PipelineMultisampleStateCreateInfo::builder()
								.rasterization_samples(SampleCountFlags::TYPE_1),
						)
						.color_blend_state(
							&PipelineColorBlendStateCreateInfo::builder().attachments(&[
								PipelineColorBlendAttachmentState::builder()
									.color_write_mask(
										ColorComponentFlags::R
											| ColorComponentFlags::G | ColorComponentFlags::B
											| ColorComponentFlags::A,
									)
									.blend_enable(true)
									.src_color_blend_factor(BlendFactor::ONE)
									.dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
									.color_blend_op(BlendOp::ADD)
									.src_alpha_blend_factor(BlendFactor::ONE_MINUS_DST_ALPHA)
									.dst_alpha_blend_factor(BlendFactor::ONE)
									.alpha_blend_op(BlendOp::ADD)
									.build(),
							]),
						)
						.dynamic_state(
							&PipelineDynamicStateCreateInfo::builder()
								.dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]),
						)
						.layout(layout)
						.push_next(
							&mut PipelineRenderingCreateInfo::builder().color_attachment_formats(&[output_format]),
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
			pool: CommandPool::new(device, *device.queue_families().graphics())?,
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

		if self.staging.poll(device).unwrap() {
			unsafe {
				self.pool.reset(device).unwrap();
			}
		}

		let arena = frame.arena();
		let mut pass = frame.pass("UI Render");
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
				view_type: ImageViewType::TYPE_2D,
				aspect: ImageAspectFlags::COLOR,
			},
		);

		for ext in self.generate_images(device, arena, delta) {
			pass.output(
				ext,
				ImageUsage {
					format: Format::R8G8B8A8_SRGB,
					usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
					view_type: ImageViewType::TYPE_2D,
					aspect: ImageAspectFlags::COLOR,
				},
			);
		}

		pass.build(move |ctx| unsafe { self.execute(ctx, PassIO { vertex, index, out }, &tris, &screen) });
	}

	pub unsafe fn destroy(self, device: &Device) {
		self.staging.destroy(device);
		self.pool.destroy(device);
		for (_, (image, view, _)) in self.images {
			view.destroy(device);
			image.destroy(device);
		}
		for (_, sampler) in self.samplers {
			sampler.destroy(device);
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
			&RenderingInfo::builder()
				.render_area(
					Rect2D::builder()
						.extent(
							Extent2D::builder()
								.width(screen.physical_size.x)
								.height(screen.physical_size.y)
								.build(),
						)
						.build(),
				)
				.layer_count(1)
				.color_attachments(&[RenderingAttachmentInfo::builder()
					.image_view(out.view)
					.image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(AttachmentLoadOp::CLEAR)
					.clear_value(ClearValue {
						color: ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(AttachmentStoreOp::STORE)
					.build()]),
		);
		ctx.device.device().cmd_set_viewport(
			ctx.buf,
			0,
			&[Viewport {
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
			.cmd_bind_pipeline(ctx.buf, PipelineBindPoint::GRAPHICS, self.pipeline);
		ctx.device.device().cmd_bind_descriptor_sets(
			ctx.buf,
			PipelineBindPoint::GRAPHICS,
			self.layout,
			0,
			&[ctx.device.base_descriptors().set()],
			&[],
		);
		ctx.device.device().cmd_push_constants(
			ctx.buf,
			self.layout,
			ShaderStageFlags::VERTEX,
			0,
			bytes_of(&PushConstantsStatic {
				screen_size: screen.physical_size,
				vertex_buffer: vertex.id.unwrap(),
			}),
		);
		ctx.device
			.device()
			.cmd_bind_index_buffer(ctx.buf, index.buffer, 0, IndexType::UINT32);

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
						&[Rect2D {
							extent: Extent2D {
								width: rect.width,
								height: rect.height,
							},
							offset: Offset2D {
								x: rect.x as _,
								y: rect.y as _,
							},
						}],
					);
					let (_, image, sampler) = &self.images[&m.texture_id];
					ctx.device.device().cmd_push_constants(
						ctx.buf,
						self.layout,
						ShaderStageFlags::FRAGMENT,
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

	fn generate_images<'a>(
		&mut self, device: &Device, arena: &'a Arena, delta: TexturesDelta,
	) -> Vec<ExternalImage, &'a Arena> {
		let mut post_barriers = Vec::new_in(arena);

		if !delta.set.is_empty() {
			self.staging
				.stage(
					device,
					self.snapshot.as_submit_info().into_iter().collect_in(arena),
					|stage| {
						let mut pre_barriers = Vec::new_in(arena);

						for (id, data) in delta.set {
							let (image, ..) = self.images.entry(id).or_insert_with(|| {
								let image = Image::create(
									device,
									ImageDesc {
										format: Format::R8G8B8A8_SRGB,
										size: Extent3D {
											width: data.image.width() as u32,
											height: data.image.height() as u32,
											depth: 1,
										},
										levels: 1,
										layers: 1,
										samples: SampleCountFlags::TYPE_1,
										usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
										..Default::default()
									},
								)
								.unwrap();

								let view = ImageView::create(
									device,
									ImageViewDesc {
										image: image.handle(),
										view_type: ImageViewType::TYPE_2D,
										format: Format::R8G8B8A8_SRGB,
										usage: ImageViewUsage::Sampled,
										aspect: ImageAspectFlags::COLOR,
									},
								)
								.unwrap();

								fn map_filter(filter: TextureFilter) -> Filter {
									match filter {
										TextureFilter::Nearest => Filter::NEAREST,
										TextureFilter::Linear => Filter::LINEAR,
									}
								}

								let sampler = self.samplers.entry(data.options).or_insert_with(|| {
									Sampler::create(
										device,
										SamplerDesc {
											mag_filter: map_filter(data.options.magnification),
											min_filter: map_filter(data.options.minification),
											mipmap_mode: SamplerMipmapMode::LINEAR,
											..Default::default()
										},
									)
									.unwrap()
								});

								(image, view, sampler.id.unwrap())
							});

							let pos = data.pos.unwrap_or([0, 0]);
							let vec: Vec<Color32, &Arena>;
							let image_subresource = ImageSubresourceLayers {
								aspect_mask: ImageAspectFlags::COLOR,
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
									image_offset: Offset3D {
										x: pos[0] as i32,
										y: pos[1] as i32,
										z: 0,
									},
									image_extent: Extent3D {
										width: data.image.width() as u32,
										height: data.image.height() as u32,
										depth: 1,
									},
								},
								*device.queue_families().graphics(),
								ImageLayout::SHADER_READ_ONLY_OPTIMAL,
								ImageLayout::SHADER_READ_ONLY_OPTIMAL,
							)?;

							if device.needs_queue_ownership_transfer() {
								let range = ImageSubresourceRange {
									aspect_mask: image_subresource.aspect_mask,
									base_mip_level: image_subresource.mip_level,
									level_count: 1,
									base_array_layer: image_subresource.base_array_layer,
									layer_count: image_subresource.layer_count,
								};
								pre_barriers.push(
									ImageMemoryBarrier2::builder()
										.image(image.handle())
										.subresource_range(range)
										.src_access_mask(AccessFlags2::SHADER_SAMPLED_READ)
										.src_stage_mask(PipelineStageFlags2::FRAGMENT_SHADER)
										.old_layout(if pos == [0, 0] {
											ImageLayout::UNDEFINED
										} else {
											ImageLayout::SHADER_READ_ONLY_OPTIMAL
										})
										.src_queue_family_index(*device.queue_families().graphics())
										.new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
										.dst_queue_family_index(*device.queue_families().transfer())
										.build(),
								);

								let (semaphore, value) = stage.wait_semaphore().unwrap();
								post_barriers.push(ExternalImage {
									handle: image.handle(),
									prev_usage: Some(ExternalSync {
										semaphore,
										value,
										queue: Some(*device.queue_families().transfer()),
										..Default::default()
									}),
									next_usage: None,
								});
							}
						}

						let buf = self.pool.next(device).unwrap();
						unsafe {
							let (semaphore, value) = stage.signal_semaphore().unwrap();

							device.device().begin_command_buffer(
								buf,
								&CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
							)?;
							device.device().cmd_pipeline_barrier2(
								buf,
								&DependencyInfo::builder().image_memory_barriers(&pre_barriers),
							);
							device.device().end_command_buffer(buf)?;
							device.submit_graphics(
								&[SubmitInfo2::builder()
									.wait_semaphore_infos(&self.snapshot.as_submit_info())
									.command_buffer_infos(&[CommandBufferSubmitInfo::builder()
										.command_buffer(buf)
										.build()])
									.signal_semaphore_infos(&[SemaphoreSubmitInfo::builder()
										.semaphore(semaphore)
										.value(value)
										.stage_mask(PipelineStageFlags2::TRANSFER)
										.build()])
									.build()],
								Fence::null(),
							)?;
						}

						Ok(())
					},
				)
				.unwrap();
		}

		for free in delta.free {
			let (image, view, _) = self.images.remove(&free).unwrap();
			unsafe {
				image.destroy(device);
				view.destroy(device);
			}
		}

		post_barriers
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
		// Transform clip rect to physical pixels:
		let clip_min_x = screen.scaling * clip_rect.min.x;
		let clip_min_y = screen.scaling * clip_rect.min.y;
		let clip_max_x = screen.scaling * clip_rect.max.x;
		let clip_max_y = screen.scaling * clip_rect.max.y;

		// Round to integer:
		let clip_min_x = clip_min_x.round() as u32;
		let clip_min_y = clip_min_y.round() as u32;
		let clip_max_x = clip_max_x.round() as u32;
		let clip_max_y = clip_max_y.round() as u32;

		// Clamp:
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
