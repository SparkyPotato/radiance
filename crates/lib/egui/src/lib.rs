#![feature(allocator_api)]

use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, cast_slice, NoUninit};
use egui::{
	epaint::{Primitive, Vertex},
	ClippedPrimitive,
	ImageData,
	Rect,
	TextureFilter,
	TextureId,
	TextureOptions,
	TextureWrapMode,
	TexturesDelta,
};
use radiance_graph::{
	arena::{Arena, IteratorAlloc},
	device::{
		descriptor::{BufferId, ImageId, SamplerId},
		Device,
	},
	graph::{
		util::{ByteReader, ImageStage},
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		ExternalImage,
		Frame,
		ImageUsage,
		ImageUsageType,
		PassContext,
		Res,
		Shader,
		VirtualResourceDesc,
	},
	resource::{BufferHandle, Image, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource, Subresource},
	util::pipeline::{default_blend, no_cull, simple_blend, GraphicsPipelineDesc},
	Result,
};
use radiance_shader_compiler::{
	c_str,
	runtime::{shader, ShaderBlob},
};
use rustc_hash::FxHashMap;
use tracing::{span, Level};
use vek::{Clamp, Vec2};

pub const SHADERS: ShaderBlob = shader!("radiance-egui");

const VERTEX_BUFFER_START_CAPACITY: u64 = (std::mem::size_of::<Vertex>() * 1024) as _;
const INDEX_BUFFER_START_CAPACITY: u64 = (std::mem::size_of::<u32>() * 1024 * 3) as _;

pub struct ScreenDescriptor {
	pub physical_size: Vec2<u32>,
	pub scaling: f32,
}

pub struct Renderer {
	images: FxHashMap<u64, (Image, Vec2<u32>, ImageView, SamplerId)>,
	samplers: FxHashMap<TextureOptions, (vk::Sampler, SamplerId)>,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	vertex_size: u64,
	index_size: u64,
}

struct PassIO {
	vertex: Res<BufferHandle>,
	index: Res<BufferHandle>,
	out: Res<ImageView>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstantsStatic {
	screen_size: Vec2<f32>,
	vertex_buffer: BufferId,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstantsDynamic {
	image: ImageId,
	sampler: SamplerId,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		let (layout, pipeline) = unsafe {
			let layout = device.device().create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
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

			let pipeline = device.graphics_pipeline(&GraphicsPipelineDesc {
				layout,
				shaders: &[
					device.shader(c_str!("radiance-egui/vertex"), vk::ShaderStageFlags::VERTEX, None),
					device.shader(c_str!("radiance-egui/pixel"), vk::ShaderStageFlags::FRAGMENT, None),
				],
				color_attachments: &[vk::Format::B8G8R8A8_UNORM],
				blend: &simple_blend(&[default_blend()]),
				raster: &no_cull(),
				..Default::default()
			})?;

			(layout, pipeline)
		};

		Ok(Self {
			images: FxHashMap::default(),
			samplers: FxHashMap::default(),
			layout,
			pipeline,
			vertex_size: VERTEX_BUFFER_START_CAPACITY,
			index_size: INDEX_BUFFER_START_CAPACITY,
		})
	}

	pub fn run<'pass, 'graph, D: VirtualResourceDesc<Resource = ImageView>>(
		&'pass mut self, frame: &mut Frame<'pass, 'graph>, tris: Vec<ClippedPrimitive>, delta: TexturesDelta,
		screen: ScreenDescriptor, out: D,
	) where
		'graph: 'pass,
	{
		let imgs = self.generate_images(frame, delta);
		let img_usage = ImageUsage {
			format: vk::Format::R8G8B8A8_UNORM,
			usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Fragment)],
			view_type: Some(vk::ImageViewType::TYPE_2D),
			subresource: Subresource::default(),
		};

		let span = span!(Level::TRACE, "setup ui pass");
		let _e = span.enter();

		let (vertices, indices) = tris
			.iter()
			.filter_map(|x| match &x.primitive {
				Primitive::Mesh(m) => Some((m.vertices.len(), m.indices.len())),
				_ => None,
			})
			.fold((0, 0), |(v1, i1), (v2, i2)| (v1 + v2, i1 + i2));

		let mut pass = frame.pass("ui");

		for x in imgs {
			pass.reference(x, img_usage);
		}

		let vertex_size = vertices * std::mem::size_of::<Vertex>();
		if vertex_size as u64 > self.vertex_size {
			self.vertex_size *= 2;
		}
		let vertex = pass.resource(
			BufferDesc {
				size: self.vertex_size,
				upload: true,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);

		let index_size = indices * std::mem::size_of::<u32>();
		if index_size as u64 > self.index_size {
			self.index_size *= 2;
		}
		let index = pass.resource(
			BufferDesc {
				size: self.index_size,
				upload: true,
			},
			BufferUsage {
				usages: &[BufferUsageType::IndexBuffer],
			},
		);
		let out = pass.resource(
			out,
			ImageUsage {
				format: vk::Format::B8G8R8A8_UNORM,
				usages: &[ImageUsageType::ColorAttachmentWrite],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);

		unsafe {
			for tris in tris.iter() {
				match &tris.primitive {
					Primitive::Mesh(m) => match m.texture_id {
						TextureId::User(x) => pass.reference::<ImageView>(Res::from_raw(x as _), img_usage),
						_ => {},
					},
					_ => {},
				}
			}
		}

		pass.build(move |ctx| unsafe { self.execute(ctx, PassIO { vertex, index, out }, &tris, &screen) });
	}

	/// # Safety
	/// Appropriate synchronization must be performed.
	pub unsafe fn destroy(self, device: &Device) {
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
		&mut self, mut pass: PassContext, io: PassIO, tris: &[ClippedPrimitive], screen: &ScreenDescriptor,
	) {
		let vertex = pass.get(io.vertex);
		let index = pass.get(io.index);
		let out = pass.get(io.out);

		Self::generate_buffers(vertex, index, tris);

		pass.device.device().cmd_begin_rendering(
			pass.buf,
			&vk::RenderingInfo::default()
				.render_area(
					vk::Rect2D::default().extent(
						vk::Extent2D::default()
							.width(screen.physical_size.x)
							.height(screen.physical_size.y),
					),
				)
				.layer_count(1)
				.color_attachments(&[vk::RenderingAttachmentInfo::default()
					.image_view(out.view)
					.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.clear_value(vk::ClearValue {
						color: vk::ClearColorValue {
							float32: [0.0, 0.0, 0.0, 1.0],
						},
					})
					.store_op(vk::AttachmentStoreOp::STORE)]),
		);
		pass.device.device().cmd_set_viewport(
			pass.buf,
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
		pass.device
			.device()
			.cmd_bind_pipeline(pass.buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
		pass.device.device().cmd_bind_descriptor_sets(
			pass.buf,
			vk::PipelineBindPoint::GRAPHICS,
			self.layout,
			0,
			&[pass.device.descriptors().set()],
			&[],
		);
		pass.device.device().cmd_push_constants(
			pass.buf,
			self.layout,
			vk::ShaderStageFlags::VERTEX,
			0,
			bytes_of(&PushConstantsStatic {
				screen_size: screen.physical_size.map(|x| x as f32) / screen.scaling,
				vertex_buffer: vertex.id.unwrap(),
			}),
		);
		pass.device
			.device()
			.cmd_bind_index_buffer(pass.buf, index.buffer, 0, vk::IndexType::UINT32);

		let mut start_index = 0;
		for prim in tris {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let rect = ScissorRect::new(&prim.clip_rect, screen);
					if rect.extent.x == 0 || rect.extent.y == 0 {
						continue;
					}

					pass.device.device().cmd_set_scissor(
						pass.buf,
						0,
						&[vk::Rect2D {
							extent: vk::Extent2D {
								width: rect.extent.x,
								height: rect.extent.y,
							},
							offset: vk::Offset2D {
								x: rect.pos.x as _,
								y: rect.pos.y as _,
							},
						}],
					);
					let (image, sampler) = match m.texture_id {
						TextureId::Managed(x) => {
							let (_, _, image, sampler) = &self.images[&x];
							(image.id.unwrap(), *sampler)
						},
						TextureId::User(x) => {
							let image: ImageView = pass.get(Res::from_raw(x as _));
							let sampler = self.samplers[&TextureOptions {
								magnification: TextureFilter::Linear,
								minification: TextureFilter::Linear,
								wrap_mode: TextureWrapMode::ClampToEdge,
							}]
								.1;
							(image.id.unwrap(), sampler)
						},
					};
					pass.device.device().cmd_push_constants(
						pass.buf,
						self.layout,
						vk::ShaderStageFlags::FRAGMENT,
						std::mem::size_of::<PushConstantsStatic>() as u32,
						bytes_of(&PushConstantsDynamic { image, sampler }),
					);
					pass.device
						.device()
						.cmd_draw_indexed(pass.buf, m.indices.len() as u32, 1, start_index, 0, 0);
					start_index += m.indices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}

		pass.device.device().cmd_end_rendering(pass.buf);
	}

	unsafe fn generate_buffers(mut vertex: BufferHandle, mut index: BufferHandle, tris: &[ClippedPrimitive]) {
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

	fn generate_images<'pass, 'graph>(
		&mut self, frame: &mut Frame<'pass, 'graph>, delta: TexturesDelta,
	) -> Vec<Res<ImageView>, &'graph Arena>
	where
		'graph: 'pass,
	{
		let mut imgs = Vec::new_in(frame.arena());
		if !delta.set.is_empty() {
			for (id, data) in delta.set {
				let x = match id {
					TextureId::Managed(x) => x,
					TextureId::User(_) => continue,
				};
				let (image, ..) = self.images.entry(x).or_insert_with(|| {
					let size = Vec2::new(data.image.width() as u32, data.image.height() as _);
					let extent = vk::Extent3D {
						width: size.x,
						height: size.y,
						depth: 1,
					};
					let image = Image::create(
						frame.device(),
						ImageDesc {
							name: "egui image",
							format: vk::Format::R8G8B8A8_UNORM,
							size: extent,
							levels: 1,
							layers: 1,
							samples: vk::SampleCountFlags::TYPE_1,
							usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
							flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
						},
					)
					.unwrap();

					let view = ImageView::create(
						frame.device(),
						ImageViewDesc {
							name: "egui image",
							image: image.handle(),
							size: extent,
							view_type: vk::ImageViewType::TYPE_2D,
							format: vk::Format::R8G8B8A8_UNORM,
							usage: ImageViewUsage::Sampled,
							subresource: Subresource::default(),
						},
					)
					.unwrap();

					fn map_filter(filter: TextureFilter) -> vk::Filter {
						match filter {
							TextureFilter::Nearest => vk::Filter::NEAREST,
							TextureFilter::Linear => vk::Filter::LINEAR,
						}
					}

					fn map_repeat(repeat: TextureWrapMode) -> vk::SamplerAddressMode {
						match repeat {
							TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
							TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
							TextureWrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
						}
					}

					let (_, id) = self.samplers.entry(data.options).or_insert_with(|| unsafe {
						let sampler = frame
							.device()
							.device()
							.create_sampler(
								&vk::SamplerCreateInfo::default()
									.mag_filter(map_filter(data.options.magnification))
									.min_filter(map_filter(data.options.minification))
									.address_mode_u(map_repeat(data.options.wrap_mode))
									.address_mode_v(map_repeat(data.options.wrap_mode))
									.address_mode_w(map_repeat(data.options.wrap_mode))
									.mipmap_mode(vk::SamplerMipmapMode::LINEAR),
								None,
							)
							.unwrap();

						let id = frame.device().descriptors().get_sampler(frame.device(), sampler);

						(sampler, id)
					});

					(image, size, view, *id)
				});

				let pos = data.pos.unwrap_or([0, 0]);
				let stage = ImageStage {
					row_stride: 0,
					plane_stride: 0,
					subresource: Subresource::default(),
					offset: vk::Offset3D {
						x: pos[0] as i32,
						y: pos[1] as i32,
						z: 0,
					},
					extent: vk::Extent3D {
						width: data.image.width() as u32,
						height: data.image.height() as u32,
						depth: 1,
					},
				};
				let vec = match data.image {
					ImageData::Color(c) => c.pixels.to_vec_in(frame.arena()),
					ImageData::Font(f) => f.srgba_pixels(None).collect_in(frame.arena()),
				};
				let img = frame.stage_image_new(
					"upload ui image",
					ExternalImage {
						handle: image.handle(),
						layout: if data.pos.is_some() {
							vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
						} else {
							vk::ImageLayout::UNDEFINED
						},
						desc: image.desc(),
					},
					stage,
					ByteReader(vec),
				);
				imgs.push(img);
			}
		}

		for free in delta.free {
			let x = match free {
				TextureId::Managed(x) => x,
				TextureId::User(_) => continue,
			};
			let (image, _, view, _) = self.images.remove(&x).unwrap();
			frame.delete(view);
			frame.delete(image);
		}

		imgs
	}
}

struct ScissorRect {
	pos: Vec2<u32>,
	extent: Vec2<u32>,
}

impl ScissorRect {
	fn new(clip_rect: &Rect, screen: &ScreenDescriptor) -> Self {
		let min = Vec2::new(clip_rect.min.x, clip_rect.min.y);
		let max = Vec2::new(clip_rect.max.x, clip_rect.max.y);

		let min = (min * screen.scaling)
			.round()
			.map(|x| x as u32)
			.clamped(Vec2::broadcast(0), screen.physical_size);
		let max = (max * screen.scaling)
			.round()
			.map(|x| x as u32)
			.clamped(min, screen.physical_size);

		Self {
			pos: min,
			extent: max - min,
		}
	}
}

pub fn to_texture_id(r: Res<ImageView>) -> TextureId { TextureId::User(r.into_raw() as _) }
