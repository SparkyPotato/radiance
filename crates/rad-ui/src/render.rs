use std::{collections::hash_map::Entry, io::Write};

use ash::vk;
use bytemuck::{bytes_of, cast_slice, NoUninit};
use egui::{
	epaint::{ImageDelta, Primitive, Vertex},
	ClippedPrimitive,
	ImageData,
	Rect,
	TextureFilter,
	TextureId,
	TextureOptions,
	TextureWrapMode,
	TexturesDelta,
};
use rad_graph::{
	arena::{Arena, IteratorAlloc},
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		GraphicsPipelineDesc,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{
		self,
		util::{ByteReader, ImageStage},
		BufferDesc,
		BufferLoc,
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
	resource::{
		BufferHandle,
		GpuPtr,
		Image,
		ImageDesc,
		ImageView,
		ImageViewDesc,
		ImageViewUsage,
		Resource,
		Subresource,
	},
	util::{
		pass::{Attachment, Load},
		pipeline::{default_blend, no_cull, simple_blend},
		render::RenderPass,
	},
	Result,
};
use rustc_hash::FxHashMap;
use tracing::{span, Level};
use vek::{Clamp, Vec2};

const VERTEX_BUFFER_START_CAPACITY: u64 = (std::mem::size_of::<Vertex>() * 1024) as _;
const INDEX_BUFFER_START_CAPACITY: u64 = (std::mem::size_of::<u32>() * 1024 * 3) as _;

pub struct ScreenDescriptor {
	pub physical_size: Vec2<u32>,
	pub scaling: f32,
}

pub struct Renderer {
	images: FxHashMap<u64, (Image, Vec2<u32>, ImageView, SamplerId)>,
	pass: RenderPass<PushConstantsStatic>,
	vertex_size: u64,
	index_size: u64,
	default_sampler: SamplerId,
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
	vertex_buffer: GpuPtr<egui::epaint::Vertex>,
}

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
struct PushConstantsDynamic {
	image: ImageId,
	sampler: SamplerId,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		let pass = RenderPass::new(
			device,
			GraphicsPipelineDesc {
				shaders: &[
					ShaderInfo {
						shader: "egui.vertex",
						..Default::default()
					},
					ShaderInfo {
						shader: "egui.pixel",
						..Default::default()
					},
				],
				color_attachments: &[vk::Format::B8G8R8A8_UNORM],
				blend: simple_blend(&[default_blend()]),
				raster: no_cull(),
				..Default::default()
			},
			false,
		)?;

		Ok(Self {
			images: FxHashMap::default(),
			pass,
			vertex_size: VERTEX_BUFFER_START_CAPACITY,
			index_size: INDEX_BUFFER_START_CAPACITY,
			default_sampler: device.sampler(SamplerDesc::default()),
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
		while vertex_size as u64 > self.vertex_size {
			self.vertex_size *= 2;
		}
		let vertex = pass.resource(
			BufferDesc {
				size: self.vertex_size,
				loc: BufferLoc::Upload,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Vertex)],
			},
		);

		let index_size = indices * std::mem::size_of::<u32>();
		while index_size as u64 > self.index_size {
			self.index_size *= 2;
		}
		let index = pass.resource(
			BufferDesc {
				size: self.index_size,
				loc: BufferLoc::Upload,
				persist: None,
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
						TextureId::User(x) if (x & (1 << 63)) == 0 => {
							pass.reference::<ImageView>(Res::from_raw(x as _), img_usage)
						},
						_ => {},
					},
					_ => {},
				}
			}
		}

		pass.build(move |pass| self.execute(pass, PassIO { vertex, index, out }, &tris, &screen));
	}

	/// # Safety
	/// Appropriate synchronization must be performed.
	pub unsafe fn destroy(self, device: &Device) {
		for (_, (image, _, view, _)) in self.images {
			view.destroy(device);
			image.destroy(device);
		}
		self.pass.destroy();
	}

	fn execute(&mut self, mut pass: PassContext, io: PassIO, tris: &[ClippedPrimitive], screen: &ScreenDescriptor) {
		let vertex = pass.get(io.vertex);
		let index = pass.get(io.index);
		unsafe {
			Self::generate_buffers(vertex, index, tris);
		}

		let mut pass = self.pass.start(
			&mut pass,
			&PushConstantsStatic {
				screen_size: screen.physical_size.map(|x| x as f32) / screen.scaling,
				vertex_buffer: vertex.ptr(),
			},
			&[Attachment {
				image: io.out,
				load: Load::Clear(vk::ClearValue {
					color: vk::ClearColorValue {
						float32: [0.0, 0.0, 0.0, 1.0],
					},
				}),
				store: true,
			}],
			None,
		);

		pass.bind_index_res(io.index, 0, vk::IndexType::UINT32);

		let mut start_index = 0;
		for prim in tris {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let rect = scissor(&prim.clip_rect, screen);
					if rect.extent.width == 0 || rect.extent.height == 0 {
						continue;
					}
					pass.scissor(rect);

					let (image, sampler) = match m.texture_id {
						TextureId::Managed(x) => {
							let (_, _, image, sampler) = &self.images[&x];
							(image.id.unwrap(), *sampler)
						},
						TextureId::User(x) => {
							let masked = x & !(1 << 63);
							let image = unsafe {
								if masked != x {
									ImageId::from_raw(masked as _)
								} else {
									pass.pass.get(Res::<ImageView>::from_raw(masked as _)).id.unwrap()
								}
							};
							(image, self.default_sampler)
						},
					};
					pass.push(
						std::mem::size_of::<PushConstantsStatic>(),
						&PushConstantsDynamic { image, sampler },
					);
					pass.draw_indexed(m.indices.len() as u32, 1, start_index, 0);
					start_index += m.indices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}
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
					vertex_slice.write_all(cast_slice(&m.vertices)).unwrap();

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

				let (handle, desc) = self.get_image_for(frame, x, &data);
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
						handle,
						layout: if data.pos.is_some() {
							vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
						} else {
							vk::ImageLayout::UNDEFINED
						},
						desc,
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

	fn get_image_for(&mut self, frame: &mut Frame, id: u64, delta: &ImageDelta) -> (vk::Image, graph::ImageDesc) {
		let size: Vec2<usize> = Vec2::from(delta.image.size()) + Vec2::from(delta.pos.unwrap_or([0; 2]));
		let size = size.map(|x| x as u32);
		match self.images.entry(id) {
			Entry::Occupied(mut x) => {
				let curr_size = x.get().1;
				if Vec2::max(curr_size, size) != curr_size {
					let (image, _, view, _) = x.insert(Self::make_image(frame.device(), size, delta.options));
					frame.delete(image);
					frame.delete(view);
				}
				let i = &x.get().0;
				(i.handle(), i.desc())
			},
			Entry::Vacant(x) => {
				let i = x.insert(Self::make_image(frame.device(), size, delta.options));
				(i.0.handle(), i.0.desc())
			},
		}
	}

	fn make_image(device: &Device, size: Vec2<u32>, opts: TextureOptions) -> (Image, Vec2<u32>, ImageView, SamplerId) {
		let extent = vk::Extent3D {
			width: size.x,
			height: size.y,
			depth: 1,
		};
		let image = Image::create(
			device,
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
			device,
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

		let id = Self::get_sampler(device, opts);

		(image, size, view, id)
	}

	fn get_sampler(device: &Device, opts: TextureOptions) -> SamplerId {
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

		device.sampler(SamplerDesc {
			mag_filter: map_filter(opts.magnification),
			min_filter: map_filter(opts.minification),
			mipmap_mode: match opts.mipmap_mode {
				Some(TextureFilter::Nearest) | None => vk::SamplerMipmapMode::NEAREST,
				Some(TextureFilter::Linear) => vk::SamplerMipmapMode::LINEAR,
			},
			address_mode_u: map_repeat(opts.wrap_mode),
			address_mode_v: map_repeat(opts.wrap_mode),
			address_mode_w: map_repeat(opts.wrap_mode),
			..Default::default()
		})
	}
}

fn scissor(clip_rect: &Rect, screen: &ScreenDescriptor) -> vk::Rect2D {
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

	let pos = min;
	let extent = max - min;
	vk::Rect2D {
		extent: vk::Extent2D {
			width: extent.x,
			height: extent.y,
		},
		offset: vk::Offset2D {
			x: pos.x as _,
			y: pos.y as _,
		},
	}
}

pub fn to_texture_id(r: Res<ImageView>) -> TextureId { TextureId::User(r.into_raw() as _) }

pub fn raw_texture_to_id(r: ImageId) -> TextureId { TextureId::User((1 << 63) | r.get() as u64) }
