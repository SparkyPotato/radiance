//! Utilities for importing assets.

use std::{
	fs::File,
	io::BufReader,
	path::Path,
	sync::atomic::{AtomicUsize, Ordering},
};

use bytemuck::{from_bytes, NoUninit};
use gltf::{
	accessor::{DataType, Dimensions},
	buffer,
	camera::Projection,
	image,
	Document,
	Gltf,
};
use meshopt::VertexDataAdapter;
use rayon::prelude::*;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Aabb, Mat4, Vec2, Vec3, Vec4};

use crate::{
	image::{Format, Image},
	material::{AlphaMode, Material},
	mesh::{Mesh, Meshlet, SubMesh, Vertex},
	scene,
	scene::{Camera, Node, Scene},
	Asset,
	AssetHeader,
	AssetMetadata,
	AssetSink,
	AssetSystem,
	AssetType,
};

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct ImportProgress {
	images: usize,
	meshes: usize,
	materials: usize,
	scenes: usize,
}

impl ImportProgress {
	pub fn as_percentage(self, total: Self) -> f32 {
		let total = total.images + total.meshes + total.materials + total.scenes;
		let self_ = self.images + self.meshes + self.materials + self.scenes;
		(self_ as f32 / total as f32) * 100.0
	}
}

pub trait ImportContext: Send + Sync {
	type Sink: AssetSink + Send;
	type Error: Send;

	/// Create a new asset sink.
	///
	/// - `name` is the name of the asset.
	/// - `header` is the header of the asset.
	fn asset(&mut self, name: &str, header: AssetHeader) -> Result<Self::Sink, Self::Error>;

	/// Report the progress of the import.
	fn progress(&self, _progress: ImportProgress, _total: ImportProgress) {}
}

#[derive(Debug)]
pub enum ImportError<S, C> {
	Gltf(gltf::Error),
	Sink(S),
	Ctx(C),
	InvalidGltf,
	UnsupportedFeature,
}

impl<S, C> ImportError<S, C> {
	fn map_ignore<T, U>(self) -> ImportError<T, U> {
		match self {
			ImportError::Gltf(err) => ImportError::Gltf(err),
			ImportError::InvalidGltf => ImportError::InvalidGltf,
			ImportError::UnsupportedFeature => ImportError::UnsupportedFeature,
			ImportError::Sink(_) | ImportError::Ctx(_) => unreachable!(),
		}
	}
}

impl<S, C> From<gltf::Error> for ImportError<S, C> {
	fn from(err: gltf::Error) -> Self { ImportError::Gltf(err) }
}

impl<S> AssetSystem<S>
where
	S: AssetSink + Send,
	S::Error: Send,
{
	/// Import an asset from a file.
	///
	/// Uses the global rayon thread pool.
	pub fn import<I: ImportContext<Sink = S>>(
		&self, mut ctx: I, path: &Path,
	) -> Result<(), ImportError<S::Error, I::Error>> {
		let s = span!(Level::INFO, "import", path = path.to_string_lossy().as_ref());
		let _e = s.enter();

		let c = &mut ctx;
		type Error<S, I> = ImportError<<S as AssetSink>::Error, <I as ImportContext>::Error>;

		let imp = {
			let s = span!(Level::TRACE, "load GLTF");
			let _e = s.enter();
			let base = path.parent().unwrap_or_else(|| Path::new("."));
			let Gltf { document: gltf, blob } =
				Gltf::from_reader(BufReader::new(File::open(path).map_err(gltf::Error::Io)?))?;
			Importer::new(base, gltf, blob).map_err(|x| x.map_ignore())?
		};

		let total = ImportProgress {
			images: imp.gltf.images().count(),
			meshes: imp.gltf.meshes().count(),
			materials: imp.gltf.materials().count(),
			scenes: imp.gltf.scenes().count(),
		};
		c.progress(ImportProgress::default(), total);

		// Images.
		let progress = AtomicUsize::new(0);
		let images: Vec<_> = imp
			.gltf
			.images()
			.map(|image| {
				let uuid = Uuid::new_v4();
				let name = image.name().unwrap_or("unnamed image");
				let header = AssetHeader {
					uuid,
					ty: AssetType::Image,
				};
				let sink = c.asset(name, header).map_err(|x| Error::<S, I>::Ctx(x))?;

				Ok::<_, Error<S, I>>((header, image, sink))
			})
			.collect();
		let images: Vec<_> = images
			.into_par_iter()
			.map(|res| {
				let (header, image, sink) = res?;
				let image = imp.image(image).map_err(|x| x.map_ignore())?;
				sink.write_data(&Asset::Image(image).to_bytes())
					.map_err(|x| ImportError::Sink(x))?;

				let old = progress.fetch_add(1, Ordering::Relaxed);
				c.progress(
					ImportProgress {
						images: old + 1,
						materials: 0,
						meshes: 0,
						scenes: 0,
					},
					total,
				);
				Ok::<_, Error<S, I>>((header, sink))
			})
			.collect::<Result<_, _>>()?;
		let images: Vec<_> = images
			.into_iter()
			.map(|(header, sink)| {
				self.assets.insert(header.uuid, AssetMetadata { header, source: sink });
				header.uuid
			})
			.collect();

		// Materials
		let progress = AtomicUsize::new(0);
		let materials: Vec<_> = imp
			.gltf
			.materials()
			.map(|material| {
				let name = material.name().unwrap_or("unnamed material");
				let uuid = Uuid::new_v4();
				let header = AssetHeader {
					uuid,
					ty: AssetType::Material,
				};
				let sink = c.asset(&name, header).map_err(|x| Error::<S, I>::Ctx(x))?;

				let material = imp.material(material, &images).map_err(|x| x.map_ignore())?;
				sink.write_data(&Asset::Material(material).to_bytes())
					.map_err(|x| ImportError::Sink(x))?;
				self.assets.insert(uuid, AssetMetadata { header, source: sink });

				let old = progress.fetch_add(1, Ordering::Relaxed);
				c.progress(
					ImportProgress {
						images: total.images,
						materials: old + 1,
						meshes: 0,
						scenes: 0,
					},
					total,
				);
				Ok::<_, Error<S, I>>(uuid)
			})
			.collect::<Result<_, _>>()?;

		// Meshes
		let progress = AtomicUsize::new(0);
		let meshes: Vec<_> = imp
			.gltf
			.meshes()
			.map(move |mesh| {
				let uuid = Uuid::new_v4();
				let name = mesh.name().unwrap_or("unnamed mesh");
				let sink = c
					.asset(
						&name,
						AssetHeader {
							uuid,
							ty: AssetType::Mesh,
						},
					)
					.map_err(|x| Error::<S, I>::Ctx(x))?;
				Ok::<_, Error<S, I>>((name, mesh, uuid, sink))
			})
			.collect::<Result<_, _>>()?;
		let meshes: Vec<_> = meshes
			.into_par_iter()
			.map(|(name, mesh, uuid, sink)| {
				let mesh = imp.mesh(&name, mesh, &materials).map_err(|x| x.map_ignore())?;
				sink.write_data(&Asset::Mesh(mesh).to_bytes())
					.map_err(|x| ImportError::Sink(x))?;
				let old = progress.fetch_add(1, Ordering::Relaxed);
				ctx.progress(
					ImportProgress {
						images: total.images,
						meshes: old + 1,
						materials: total.materials,
						scenes: 0,
					},
					total,
				);
				Ok::<_, ImportError<S::Error, I::Error>>((uuid, sink))
			})
			.collect::<Result<_, _>>()?;
		let meshes: Vec<_> = meshes
			.into_iter()
			.map(|(uuid, sink)| {
				self.assets.insert(
					uuid,
					AssetMetadata {
						header: AssetHeader {
							uuid,
							ty: AssetType::Mesh,
						},
						source: sink,
					},
				);
				uuid
			})
			.collect();

		for scene in imp.gltf.scenes() {
			let name = scene.name().unwrap_or("unnamed scene");
			let i = scene.index();
			let out = imp.scene(&name, scene, &meshes).map_err(|x| x.map_ignore())?;
			let header = AssetHeader {
				uuid: Uuid::new_v4(),
				ty: AssetType::Scene,
			};
			let sink = ctx.asset(&name, header).map_err(|x| Error::<S, I>::Ctx(x))?;
			sink.write_data(&Asset::Scene(out).to_bytes())
				.map_err(|x| ImportError::Sink(x))?;
			self.assets.insert(header.uuid, AssetMetadata { header, source: sink });
			ctx.progress(
				ImportProgress {
					images: total.images,
					meshes: total.meshes,
					materials: total.materials,
					scenes: i + 1,
				},
				total,
			);
		}

		Ok(())
	}
}

type ImportResult<T> = Result<T, ImportError<(), ()>>;

struct Importer<'a> {
	base: &'a Path,
	gltf: Document,
	buffers: Vec<buffer::Data>,
}

impl<'a> Importer<'a> {
	fn new(base: &'a Path, gltf: Document, mut blob: Option<Vec<u8>>) -> ImportResult<Self> {
		let buffers = gltf
			.buffers()
			.map(|buffer| {
				let data = buffer::Data::from_source_and_blob(buffer.source(), Some(base), &mut blob)?;
				if data.len() < buffer.length() {
					return Err(gltf::Error::BufferLength {
						buffer: buffer.index(),
						expected: buffer.length(),
						actual: data.len(),
					});
				}
				Ok(data)
			})
			.collect::<Result<Vec<_>, _>>()?;
		Ok(Self { base, gltf, buffers })
	}

	fn image(&self, image: gltf::Image) -> ImportResult<Image> {
		let name = image.name().unwrap_or("unnamed image");

		let s = span!(Level::INFO, "importing image", name = name);
		let _e = s.enter();

		let mut image = image::Data::from_source(image.source(), Some(self.base), &self.buffers)?;
		let format = match image.format {
			image::Format::R8 => Format::R8,
			image::Format::R8G8 => Format::R8G8,
			image::Format::R8G8B8 => {
				let p = image.pixels;
				image.pixels = Vec::with_capacity(p.len() / 3 * 4);
				for i in 0..(p.len() / 3) {
					image.pixels.push(p[i * 3]);
					image.pixels.push(p[i * 3 + 1]);
					image.pixels.push(p[i * 3 + 2]);
					image.pixels.push(255);
				}
				Format::R8G8B8A8
			},
			image::Format::R8G8B8A8 => Format::R8G8B8A8,
			image::Format::R16 => Format::R16,
			image::Format::R16G16 => Format::R16G16,
			image::Format::R16G16B16 => Format::R16G16B16,
			image::Format::R16G16B16A16 => Format::R16G16B16A16,
			image::Format::R32G32B32FLOAT => Format::R32G32B32FLOAT,
			image::Format::R32G32B32A32FLOAT => Format::R32G32B32A32FLOAT,
		};

		Ok(Image {
			width: image.width,
			height: image.height,
			format,
			data: image.pixels,
		})
	}

	fn material(&self, material: gltf::Material, images: &[Uuid]) -> ImportResult<Material> {
		let name = material.name().unwrap_or("unnamed material");

		let s = span!(Level::INFO, "importing material", name = name);
		let _e = s.enter();

		let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
		let alpha_mode = match material.alpha_mode() {
			gltf::material::AlphaMode::Blend => AlphaMode::Blend,
			gltf::material::AlphaMode::Mask => AlphaMode::Mask,
			gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
		};

		let pbr = material.pbr_metallic_roughness();
		let base_color_factor = pbr.base_color_factor();
		let base_color = pbr.base_color_texture().map(|x| images[x.texture().source().index()]);
		let metallic_factor = pbr.metallic_factor();
		let roughness_factor = pbr.roughness_factor();
		let metallic_roughness = pbr
			.metallic_roughness_texture()
			.map(|x| images[x.texture().source().index()]);
		let normal = material.normal_texture().map(|x| images[x.texture().source().index()]);
		let occlusion = material
			.occlusion_texture()
			.map(|x| images[x.texture().source().index()]);
		let emissive_factor = material.emissive_factor();
		let emissive = material
			.emissive_texture()
			.map(|x| images[x.texture().source().index()]);

		Ok(Material {
			alpha_cutoff,
			alpha_mode,
			base_color_factor: Vec4 {
				x: base_color_factor[0],
				y: base_color_factor[1],
				z: base_color_factor[2],
				w: base_color_factor[3],
			},
			base_color,
			metallic_factor,
			roughness_factor,
			metallic_roughness,
			normal,
			occlusion,
			emissive_factor: Vec3 {
				x: emissive_factor[0],
				y: emissive_factor[1],
				z: emissive_factor[2],
			},

			emissive,
		})
	}

	fn mesh(&self, name: &str, mesh: gltf::Mesh, materials: &[Uuid]) -> ImportResult<Mesh> {
		let s = span!(Level::INFO, "importing mesh", name = name);
		let _e = s.enter();

		let mut out = Mesh {
			vertices: Vec::new(),
			indices: Vec::new(),
			meshlets: Vec::new(),
			submeshes: Vec::new(),
			aabb: Aabb {
				min: Vec3::broadcast(f32::INFINITY),
				max: Vec3::broadcast(f32::NEG_INFINITY),
			},
		};
		for prim in mesh.primitives() {
			// Goofy GLTF things.
			let indices = prim.indices().ok_or(ImportError::InvalidGltf)?;
			let (indices, ty, comp) = self.accessor(indices)?;
			if comp != Dimensions::Scalar {
				return Err(ImportError::InvalidGltf);
			}
			let indices: Vec<_> = match ty {
				DataType::U8 => indices.flatten().map(|&i| i as u32).collect(),
				DataType::U16 => indices.map(|i| *from_bytes::<u16>(i) as u32).collect(),
				DataType::U32 => indices.map(|i| *from_bytes::<u32>(i)).collect(),
				_ => return Err(ImportError::InvalidGltf),
			};

			let positions = prim.get(&gltf::Semantic::Positions).ok_or(ImportError::InvalidGltf)?;
			let (positions, ty, comp) = self.accessor(positions)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(ImportError::InvalidGltf);
			}
			let positions = positions.map(|p| *from_bytes::<Vec3<f32>>(p));

			let normals = prim.get(&gltf::Semantic::Normals).ok_or(ImportError::InvalidGltf)?;
			let (normals, ty, comp) = self.accessor(normals)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(ImportError::InvalidGltf);
			}
			let normals = normals.map(|n| *from_bytes::<Vec3<f32>>(n));

			let uv = prim.get(&gltf::Semantic::TexCoords(0));
			let mut uv = uv
				.map(|uv| {
					let (uv, ty, comp) = self.accessor(uv)?;
					if comp != Dimensions::Vec2 {
						return Err(ImportError::InvalidGltf);
					}

					if !matches!(ty, DataType::F32 | DataType::U8 | DataType::U16) {
						return Err(ImportError::InvalidGltf);
					}
					Ok(uv.map(move |uv| match ty {
						DataType::F32 => *from_bytes(uv),
						DataType::U8 => from_bytes::<Vec2<u8>>(uv).map(|u| u as f32 / 255.0),
						DataType::U16 => from_bytes::<Vec2<u16>>(uv).map(|u| u as f32 / 65535.0),
						_ => panic!("yikes"),
					}))
				})
				.transpose()?;

			#[derive(Copy, Clone, Default, NoUninit)]
			#[repr(C)]
			struct TempVertex {
				position: Vec3<f32>,
				normal: Vec3<f32>,
				uv: Vec2<f32>,
			}
			let vertices: Vec<TempVertex> = positions
				.zip(normals)
				.zip(std::iter::from_fn(move || {
					if let Some(ref mut uv) = uv {
						uv.next()
					} else {
						Some(Vec2::new(0.0, 0.0))
					}
				}))
				.map(|((position, normal), uv)| TempVertex { position, normal, uv })
				.collect();

			// Optimizations and meshlet building.
			let (vertices, indices) = {
				let s = span!(Level::INFO, "optimizing submesh");
				let _e = s.enter();

				let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(&indices));
				let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap);
				let mut indices = meshopt::remap_index_buffer(Some(&indices), vertex_count, &remap);
				meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());
				meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

				(vertices, indices)
			};

			let adapter = VertexDataAdapter::new(
				bytemuck::cast_slice(vertices.as_slice()),
				std::mem::size_of::<TempVertex>(),
				0,
			)
			.unwrap();

			let meshlets = {
				let s = span!(Level::INFO, "building meshlets");
				let _e = s.enter();

				meshopt::build_meshlets(&indices, &adapter, 64, 124, 0.5)
			};

			// Build the final mesh.
			let off = out.meshlets.len() as u32;
			let mut sub = SubMesh {
				meshlets: off..(off + meshlets.len() as u32),
				aabb: Aabb {
					min: Vec3::broadcast(f32::INFINITY),
					max: Vec3::broadcast(f32::NEG_INFINITY),
				},
				material: materials[prim.material().index().unwrap()],
			};
			out.meshlets.extend(meshlets.iter().map(|m| {
				let vertices = m.vertices.iter().map(|&x| vertices[x as usize]);
				let mut aabb = Aabb {
					min: Vec3::broadcast(f32::INFINITY),
					max: Vec3::broadcast(f32::NEG_INFINITY),
				};
				for v in vertices.clone() {
					aabb.expand_to_contain_point(v.position);
				}
				sub.aabb.expand_to_contain(aabb);

				let extent = aabb.max - aabb.min;

				let index_offset = out.indices.len() as u32;
				let vertex_offset = out.vertices.len() as u32;
				let vert_count = m.vertices.len() as u8;
				let tri_count = (m.triangles.len() / 3) as u8;
				out.vertices.extend(vertices.map(|x| Vertex {
					position: ((x.position - aabb.min) / extent * Vec3::broadcast(65535.0)).map(|x| x.round() as u16),
					normal: (x.normal * Vec3::broadcast(32767.0)).map(|x| x.round() as i16),
					uv: (x.uv * Vec2::broadcast(65535.0)).map(|x| x.round() as u16),
				}));
				out.indices.extend(m.triangles);

				Meshlet {
					aabb,
					index_offset,
					vertex_offset,
					tri_count,
					vert_count,
				}
			}));
			out.aabb.expand_to_contain(sub.aabb);
			out.submeshes.push(sub);
		}

		Ok(out)
	}

	fn scene(&self, name: &str, scene: gltf::Scene, meshes: &[Uuid]) -> ImportResult<Scene> {
		let s = span!(Level::INFO, "importing scene", name = name);
		let _e = s.enter();

		let mut nodes = 0;
		let mut cameras = 0;
		for node in scene.nodes() {
			if node.mesh().is_some() {
				nodes += 1;
			}
			if node.camera().is_some() {
				cameras += 1;
			}
		}

		let mut out = Scene {
			nodes: Vec::with_capacity(nodes),
			cameras: Vec::with_capacity(cameras),
		};

		for node in scene.nodes() {
			self.node(node, Mat4::identity(), meshes, &mut out);
		}

		Ok(out)
	}

	fn node(&self, node: gltf::Node, transform: Mat4<f32>, meshes: &[Uuid], out: &mut Scene) {
		let this_transform = Mat4::from_col_arrays(node.transform().matrix());
		let transform = transform * this_transform;

		let model = node.mesh().map(|mesh| {
			let name = node.name().unwrap_or("unnamed node").to_string();
			let model = meshes[mesh.index()];

			Node { name, transform, model }
		});
		out.nodes.extend(model);

		let camera = node.camera().map(|camera| {
			let name = camera.name().unwrap_or("unnamed camera").to_string();
			let view = transform.inverted();
			let projection = match camera.projection() {
				Projection::Perspective(p) => {
					let yfov = p.yfov();
					let near = p.znear();
					let far = p.zfar();
					scene::Projection::Perspective { yfov, near, far }
				},
				Projection::Orthographic(o) => {
					let height = o.ymag();
					let near = o.znear();
					let far = o.zfar();
					scene::Projection::Orthographic { height, near, far }
				},
			};
			Camera { name, view, projection }
		});
		out.cameras.extend(camera);

		for child in node.children() {
			self.node(child, transform, meshes, out);
		}
	}

	pub fn accessor(
		&self, accessor: gltf::Accessor,
	) -> ImportResult<(impl Iterator<Item = &[u8]>, DataType, Dimensions)> {
		if accessor.sparse().is_some() {
			return Err(ImportError::UnsupportedFeature);
		}

		let ty = accessor.data_type();
		let dim = accessor.dimensions();
		let size = ty.size()
			* match dim {
				Dimensions::Scalar => 1,
				Dimensions::Vec2 => 2,
				Dimensions::Vec3 => 3,
				Dimensions::Vec4 => 4,
				Dimensions::Mat2 => 4,
				Dimensions::Mat3 => 9,
				Dimensions::Mat4 => 16,
			};

		let view = accessor.view().ok_or(ImportError::InvalidGltf)?;
		let buffer = &self.buffers[view.buffer().index()];
		let stride = view.stride().unwrap_or(size);
		let view = &buffer[view.offset()..view.offset() + view.length()];
		let offset = &view[accessor.offset()..];

		Ok((
			(0..accessor.count()).map(move |x| &offset[x * stride..][..size]),
			ty,
			dim,
		))
	}
}

