//! Utilities for importing assets.

use std::{
	fs::File,
	io::BufReader,
	path::Path,
	sync::atomic::{AtomicUsize, Ordering},
};

use gltf::{
	accessor::{DataType, Dimensions},
	buffer,
	camera::Projection,
	Document,
	Gltf,
};
use rayon::prelude::*;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Mat4, Vec3, Vec4};

use crate::{
	material::{AlphaMode, Material},
	scene,
	scene::{Camera, Node, Scene},
	Asset,
	AssetHeader,
	AssetMetadata,
	AssetSink,
	AssetSystem,
	AssetType,
};

pub mod mesh;

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct ImportProgress {
	meshes: usize,
	materials: usize,
	scenes: usize,
}

impl ImportProgress {
	pub fn as_percentage(self, total: Self) -> f32 {
		let total = total.meshes + total.materials + total.scenes;
		let self_ = self.meshes + self.materials + self.scenes;
		self_ as f32 / total as f32
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
			meshes: imp.gltf.meshes().count(),
			materials: imp.gltf.materials().count(),
			scenes: imp.gltf.scenes().count(),
		};
		c.progress(ImportProgress::default(), total);

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

				let material = imp.material(material).map_err(|x| x.map_ignore())?;
				sink.write_data(&Asset::Material(material).to_bytes())
					.map_err(|x| ImportError::Sink(x))?;
				self.assets.insert(uuid, AssetMetadata { header, source: sink });

				let old = progress.fetch_add(1, Ordering::Relaxed);
				c.progress(
					ImportProgress {
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

	fn material(&self, material: gltf::Material) -> ImportResult<Material> {
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
		let metallic_factor = pbr.metallic_factor();
		let roughness_factor = pbr.roughness_factor();
		let emissive_factor = material.emissive_factor();

		Ok(Material {
			alpha_cutoff,
			alpha_mode,
			base_color_factor: Vec4 {
				x: base_color_factor[0],
				y: base_color_factor[1],
				z: base_color_factor[2],
				w: base_color_factor[3],
			},
			metallic_factor,
			roughness_factor,
			emissive_factor: Vec3 {
				x: emissive_factor[0],
				y: emissive_factor[1],
				z: emissive_factor[2],
			},
		})
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
