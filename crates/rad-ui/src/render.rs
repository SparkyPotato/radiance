use ash::vk;
use bytemuck::NoUninit;
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
use hashbrown::hash_map::Entry;
use rad_graph::{
	arena::IteratorAlloc,
	device::{
		descriptor::{ImageId, SamplerId},
		Device,
		GraphicsPipelineDesc,
		SamplerDesc,
		ShaderInfo,
	},
	graph::{
		ArenaMap,
		BufferDesc,
		BufferUsage,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Persist,
		Res,
		Shader,
		SwapchainImage,
	},
	resource::{BufferHandle, GpuPtr, ImageView, Subresource},
	util::{
		pass::{Attachment, ImageCopy, Load},
		pipeline::{default_blend, no_cull, simple_blend},
		render::{FullscreenPass, RenderPass},
		staging::ByteReader,
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
	images: FxHashMap<u64, (Persist<ImageView>, Vec2<u32>, SamplerId)>,
	sdr: RenderPass<PushConstantsStatic>,
	hdr: RenderPass<PushConstantsStatic>,
	blit: FullscreenPass<ImageId>,
	vertex_size: u64,
	index_size: u64,
	default_sampler: SamplerId,
}

struct PassIO<'a> {
	vertex: Res<BufferHandle>,
	index: Res<BufferHandle>,
	out: Res<ImageView>,
	imgs: ArenaMap<'a, u64, Res<ImageView>>,
	hdr: bool,
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
	hdr_image: u32,
}

impl Renderer {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			images: FxHashMap::default(),
			sdr: Self::make_pass(device, false)?,
			hdr: Self::make_pass(device, true)?,
			blit: FullscreenPass::new(
				device,
				ShaderInfo {
					shader: "egui.blit",
					..Default::default()
				},
				&[vk::Format::A2B10G10R10_UNORM_PACK32],
			)?,
			vertex_size: VERTEX_BUFFER_START_CAPACITY,
			index_size: INDEX_BUFFER_START_CAPACITY,
			default_sampler: device.sampler(SamplerDesc::default()),
		})
	}

	fn make_pass(device: &Device, hdr: bool) -> Result<RenderPass<PushConstantsStatic>> {
		RenderPass::new(
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
				color_attachments: &[if hdr {
					vk::Format::R16G16B16A16_SFLOAT
				} else {
					vk::Format::B8G8R8A8_UNORM
				}],
				blend: simple_blend(&[default_blend()]),
				raster: no_cull(),
				..Default::default()
			},
			false,
		)
	}

	pub fn run<'pass, 'graph>(
		&'pass mut self, frame: &mut Frame<'pass, 'graph>, tris: Vec<ClippedPrimitive>, delta: TexturesDelta,
		screen: ScreenDescriptor, out: SwapchainImage,
	) where
		'graph: 'pass,
	{
		let mut imgs = self.generate_images(frame, delta);

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

		self.reference_images(&mut pass, &mut imgs);

		let vertex_size = vertices * std::mem::size_of::<Vertex>();
		while vertex_size as u64 > self.vertex_size {
			self.vertex_size *= 2;
		}
		let vertex = pass.resource(BufferDesc::upload(self.vertex_size), BufferUsage::read(Shader::Vertex));

		let index_size = indices * std::mem::size_of::<u32>();
		while index_size as u64 > self.index_size {
			self.index_size *= 2;
		}
		let index = pass.resource(BufferDesc::upload(self.index_size), BufferUsage::index());

		let hdr = out.format == vk::Format::A2B10G10R10_UNORM_PACK32;
		// We need real alpha, so we can't use the swapchain HDR image
		let outr = if hdr {
			pass.resource(
				ImageDesc {
					size: vk::Extent3D::default()
						.width(out.size.width)
						.height(out.size.height)
						.depth(1),
					format: vk::Format::R16G16B16A16_SFLOAT,
					..Default::default()
				},
				ImageUsage::color_attachment(),
			)
		} else {
			pass.resource(out, ImageUsage::format_color_attachment(vk::Format::B8G8R8A8_UNORM))
		};

		unsafe {
			for tris in tris.iter() {
				match &tris.primitive {
					Primitive::Mesh(m) => match m.texture_id {
						TextureId::User(x) if (x & (1 << 63)) == 0 => {
							let res = Res::<ImageView>::from_raw(x as _);
							let desc = pass.desc(res);
							pass.reference::<ImageView>(
								res,
								ImageUsage::format_sampled_2d(
									if desc.format == vk::Format::R8G8B8A8_SRGB {
										vk::Format::R8G8B8A8_UNORM
									} else {
										desc.format
									},
									Shader::Fragment,
								),
							)
						},
						_ => {},
					},
					_ => {},
				}
			}
		}

		let this = &*self;
		pass.build(move |pass| {
			this.execute(
				pass,
				PassIO {
					vertex,
					index,
					out: outr,
					imgs,
					hdr,
				},
				&tris,
				&screen,
			)
		});

		if hdr {
			let mut pass = frame.pass("blit ui");
			pass.reference(outr, ImageUsage::sampled_2d(Shader::Fragment));
			let out = pass.resource(out, ImageUsage::color_attachment());
			pass.build(move |mut pass| {
				let id = pass.get(outr).id.unwrap();
				this.blit.run_one(&mut pass, &id, out);
			});
		}
	}

	/// # Safety
	/// Appropriate synchronization must be performed.
	pub unsafe fn destroy(self) {
		self.sdr.destroy();
		self.hdr.destroy();
	}

	fn execute(&self, mut pass: PassContext, io: PassIO<'_>, tris: &[ClippedPrimitive], screen: &ScreenDescriptor) {
		Self::generate_buffers(&mut pass, io.vertex, io.index, tris);

		let vertex_buffer = pass.get(io.vertex).ptr();
		let mut pass = if io.hdr { &self.hdr } else { &self.sdr }.start(
			&mut pass,
			&PushConstantsStatic {
				screen_size: screen.physical_size.map(|x| x as f32) / screen.scaling,
				vertex_buffer,
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
		let mut start_vertex = 0;
		for prim in tris {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					let rect = scissor(&prim.clip_rect, screen);
					if rect.extent.width == 0 || rect.extent.height == 0 {
						continue;
					}
					pass.scissor(rect);

					let (image, sampler, hdr_image) = match m.texture_id {
						TextureId::Managed(x) => {
							let &img = io.imgs.get(&x).unwrap();
							let &(_, _, sampler) = self.images.get(&x).unwrap();
							(pass.pass.get(img).id.unwrap(), sampler, false)
						},
						TextureId::User(x) => {
							let masked = x & !(1 << 63);
							let (image, hdr) = unsafe {
								if masked != x {
									(ImageId::from_raw(masked as _), false)
								} else {
									let id = Res::<ImageView>::from_raw(masked as _);
									(
										pass.pass.get(id).id.unwrap(),
										pass.pass.desc(id).format == vk::Format::A2B10G10R10_UNORM_PACK32,
									)
								}
							};
							(image, self.default_sampler, hdr)
						},
					};
					pass.push(
						std::mem::size_of::<PushConstantsStatic>(),
						&PushConstantsDynamic {
							image,
							sampler,
							hdr_image: hdr_image as _,
						},
					);
					pass.draw_indexed(m.indices.len() as u32, 1, start_index, start_vertex, 0);
					start_index += m.indices.len() as u32;
					start_vertex += m.vertices.len() as u32;
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}
	}

	fn generate_buffers(
		pass: &mut PassContext, vertex: Res<BufferHandle>, index: Res<BufferHandle>, tris: &[ClippedPrimitive],
	) {
		let span = span!(Level::TRACE, "upload ui buffers");
		let _e = span.enter();

		let mut vertices = 0;
		let mut indices = 0;
		for prim in tris.iter() {
			match &prim.primitive {
				Primitive::Mesh(m) => {
					pass.write(vertex, vertices * std::mem::size_of::<Vertex>(), &m.vertices);
					pass.write(index, indices * std::mem::size_of::<u32>(), &m.indices);
					vertices += m.vertices.len();
					indices += m.indices.len();
				},
				Primitive::Callback(_) => panic!("Callback not supported"),
			}
		}
	}

	fn generate_images<'pass, 'graph>(
		&mut self, frame: &mut Frame<'pass, 'graph>, delta: TexturesDelta,
	) -> ArenaMap<'graph, u64, Res<ImageView>>
	where
		'graph: 'pass,
	{
		let mut imgs = ArenaMap::with_capacity_and_hasher_in(self.images.len(), Default::default(), frame.arena());
		for (id, data) in delta.set {
			let x = match id {
				TextureId::Managed(x) => x,
				TextureId::User(_) => continue,
			};

			let pos = data.pos.unwrap_or([0; 2]);
			let extent: Vec2<usize> = Vec2::from(data.image.size());
			let size: Vec2<usize> = extent + Vec2::from(pos);
			let size = size.map(|x| x as u32);
			let (persist, size, _) = self.images.entry(x).or_insert_with(|| {
				let sampler = Self::get_sampler(frame.device(), data.options);
				(Persist::new(), size, sampler)
			});
			let persist = *persist;
			let size = *size;
			let size = vk::Extent3D {
				width: size.x,
				height: size.y,
				depth: 1,
			};

			let vec = ByteReader(match data.image {
				ImageData::Color(c) => c.pixels.to_vec_in(frame.arena()),
				ImageData::Font(f) => f.srgba_pixels(None).collect_in(frame.arena()),
			});

			let mut pass = frame.pass("upload egui image");
			let staging = pass.resource(
				BufferDesc::staging(vec.as_ref().len() as _),
				BufferUsage::transfer_read(),
			);
			let img = pass.resource(
				ImageDesc {
					size,
					format: vk::Format::R8G8B8A8_UNORM,
					persist: Some(persist),
					..Default::default()
				},
				ImageUsage::transfer_write(),
			);
			// TODO: transfer queue
			pass.build(move |mut pass| {
				pass.write(staging, 0, vec.as_ref());
				pass.copy_buffer_to_image(
					staging,
					img,
					0,
					ImageCopy {
						row_stride: 0,
						plane_stride: 0,
						subresource: Subresource::default(),
						offset: vk::Offset3D {
							x: pos[0] as i32,
							y: pos[1] as i32,
							z: 0,
						},
						extent: vk::Extent3D {
							width: extent.x as _,
							height: extent.y as _,
							depth: 1,
						},
					},
				);
			});

			imgs.insert(x, img);
		}

		for free in delta.free {
			let x = match free {
				TextureId::Managed(x) => x,
				TextureId::User(_) => continue,
			};
			self.images.remove(&x);
		}

		imgs
	}

	fn reference_images(&self, pass: &mut PassBuilder, imgs: &mut ArenaMap<'_, u64, Res<ImageView>>) {
		let format = vk::Format::R8G8B8A8_UNORM;
		for (&id, &(image, size, _)) in self.images.iter() {
			match imgs.entry(id) {
				Entry::Occupied(x) => pass.reference(*x.get(), ImageUsage::format_sampled_2d(format, Shader::Fragment)),
				Entry::Vacant(v) => {
					v.insert(pass.resource(
						ImageDesc {
							size: vk::Extent3D {
								width: size.x,
								height: size.y,
								depth: 1,
							},
							format,
							persist: Some(image),
							..Default::default()
						},
						ImageUsage {
							format,
							usages: &[
								ImageUsageType::ShaderReadSampledImage(Shader::Fragment),
								ImageUsageType::AddUsage(vk::ImageUsageFlags::TRANSFER_DST),
							],
							view_type: Some(vk::ImageViewType::TYPE_2D),
							subresource: Subresource::default(),
						},
					));
				},
			}
		}
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
