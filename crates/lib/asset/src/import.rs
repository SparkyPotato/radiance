//! Utilities for importing assets.

use std::{
	path::Path,
	sync::atomic::{AtomicUsize, Ordering},
};

use bytemuck::NoUninit;
use gltf::{
	accessor::{DataType, Dimensions},
	buffer,
	camera::Projection,
	Accessor,
	Document,
	Primitive,
	Semantic,
};
use meshopt::VertexDataAdapter;
use rayon::prelude::*;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Aabb, Mat4, Vec2, Vec3};

use crate::{
	mesh::{Cone, Mesh, Meshlet, Vertex},
	model::Model,
	scene,
	scene::{Camera, Node, Scene},
	util::SliceReader,
	AssetHeader,
	AssetMetadata,
	AssetSink,
	AssetSystem,
	AssetType,
};

#[derive(Copy, Clone)]
pub struct ImportProgress {
	meshes: usize,
	models: usize,
	materials: usize,
	scenes: usize,
}

impl ImportProgress {
	pub fn as_percentage(self, total: Self) -> f32 {
		let total = total.meshes + total.models + total.materials + total.scenes;
		let self_ = self.meshes + self.models + self.materials + self.scenes;
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
		&mut self, mut ctx: I, path: &Path,
	) -> Result<(), ImportError<S::Error, I::Error>> {
		let s = span!(Level::INFO, "import", path = path.to_string_lossy().as_ref());
		let _e = s.enter();

		let c = &mut ctx;
		type Error<S, I> = ImportError<<S as AssetSink>::Error, <I as ImportContext>::Error>;

		let (gltf, buffers, ..) = gltf::import(path)?;
		let imp = Importer { gltf, buffers }; // images };d

		let total = ImportProgress {
			meshes: imp.gltf.meshes().flat_map(|x| x.primitives()).count(),
			models: imp.gltf.meshes().count(),
			materials: imp.gltf.materials().count(),
			scenes: imp.gltf.scenes().count(),
		};

		// Meshes
		let progress = AtomicUsize::new(0);
		let prims: Vec<_> = imp
			.gltf
			.meshes()
			.flat_map(move |mesh| {
				let name = mesh.name().unwrap_or("unnamed mesh");
				let prims = mesh.primitives();
				let mesh_id = mesh.index();
				prims.map(move |prim| (mesh_id, name, prim))
			})
			.map(move |(mesh_id, name, prim)| {
				let uuid = Uuid::new_v4();
				let sink = c
					.asset(
						&name,
						AssetHeader {
							uuid,
							ty: AssetType::Mesh,
						},
					)
					.map_err(|x| Error::<S, I>::Ctx(x))?;
				Ok::<_, Error<S, I>>((mesh_id, uuid, name, prim, sink))
			})
			.collect::<Result<_, _>>()?;
		let meshes: Vec<_> = prims
			.into_par_iter()
			.map(|(mesh_id, uuid, name, prim, mut sink)| {
				let mesh = imp.mesh(&name, prim).map_err(|x| x.map_ignore())?;
				sink.write_data(&mesh.to_bytes()).map_err(|x| ImportError::Sink(x))?;
				let old = progress.fetch_add(1, Ordering::Relaxed);
				ctx.progress(
					ImportProgress {
						meshes: old + 1,
						models: 0,
						materials: 0,
						scenes: 0,
					},
					total,
				);
				Ok::<_, ImportError<S::Error, I::Error>>((mesh_id, mesh.aabb, uuid, sink))
			})
			.collect::<Result<_, _>>()?;

		// Models
		let mut models: Vec<_> = imp
			.gltf
			.meshes()
			.map(|mesh| {
				(
					mesh.name().unwrap_or("unnamed mesh").to_string(),
					Model {
						meshes: Vec::new(),
						aabb: Aabb {
							min: Vec3::broadcast(f32::INFINITY),
							max: Vec3::broadcast(f32::NEG_INFINITY),
						},
					},
				)
			})
			.collect();
		for (mesh_id, aabb, uuid, sink) in meshes.into_iter() {
			let model = &mut models[mesh_id].1;
			model.meshes.push(uuid);
			model.aabb = model.aabb.union(aabb);
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
		}
		let models: Vec<_> = models
			.into_iter()
			.enumerate()
			.map(|(i, (name, model))| {
				let uuid = Uuid::new_v4();
				let header = AssetHeader {
					uuid,
					ty: AssetType::Model,
				};
				let mut sink = ctx.asset(&name, header).map_err(|x| Error::<S, I>::Ctx(x))?;
				sink.write_data(&model.to_bytes()).map_err(|x| ImportError::Sink(x))?;
				self.assets.insert(uuid, AssetMetadata { header, source: sink });
				ctx.progress(
					ImportProgress {
						meshes: total.meshes,
						models: i + 1,
						materials: 0,
						scenes: 0,
					},
					total,
				);
				Ok::<_, ImportError<S::Error, I::Error>>(uuid)
			})
			.collect::<Result<_, _>>()?;

		for scene in imp.gltf.scenes() {
			let name = scene.name().unwrap_or("unnamed scene");
			let i = scene.index();
			let out = imp.scene(&name, scene, &models).map_err(|x| x.map_ignore())?;
			let header = AssetHeader {
				uuid: Uuid::new_v4(),
				ty: AssetType::Scene,
			};
			let mut sink = ctx.asset(&name, header).map_err(|x| Error::<S, I>::Ctx(x))?;
			sink.write_data(&out.to_bytes()).map_err(|x| ImportError::Sink(x))?;
			self.assets.insert(header.uuid, AssetMetadata { header, source: sink });
			ctx.progress(
				ImportProgress {
					meshes: total.meshes,
					models: total.models,
					materials: 0,
					scenes: i + 1,
				},
				total,
			);
		}

		Ok(())
	}
}

type ImportResult<T> = Result<T, ImportError<(), ()>>;

struct Importer {
	gltf: Document,
	buffers: Vec<buffer::Data>,
	// images: Vec<image::Data>,
}

impl Importer {
	fn mesh(&self, name: &str, prim: Primitive) -> ImportResult<Mesh> {
		let s = span!(Level::INFO, "importing mesh", name = name);
		let _e = s.enter();

		// Goofy GLTF things.
		let indices = prim.indices().ok_or(ImportError::InvalidGltf)?;
		let (indices, ty, comp) = self.accessor(indices)?;
		if comp != Dimensions::Scalar {
			return Err(ImportError::InvalidGltf);
		}
		let indices = match ty {
			DataType::U8 => indices.iter().map(|&i| i as u32).collect(),
			DataType::U16 => bytemuck::cast_slice::<_, u16>(indices)
				.iter()
				.map(|&i| i as u32)
				.collect(),
			DataType::U32 => bytemuck::cast_slice(indices).to_vec(),
			_ => return Err(ImportError::InvalidGltf),
		};

		let positions = prim.get(&Semantic::Positions).ok_or(ImportError::InvalidGltf)?;
		let (positions, ty, comp) = self.accessor(positions)?;
		if comp != Dimensions::Vec3 || ty != DataType::F32 {
			return Err(ImportError::InvalidGltf);
		}
		let positions = bytemuck::cast_slice::<_, Vec3<f32>>(positions).iter().copied();

		let normals = prim.get(&Semantic::Normals).ok_or(ImportError::InvalidGltf)?;
		let (normals, ty, comp) = self.accessor(normals)?;
		if comp != Dimensions::Vec3 || ty != DataType::F32 {
			return Err(ImportError::InvalidGltf);
		}
		let normals = bytemuck::cast_slice::<_, Vec3<f32>>(normals).iter().copied();

		let uv = prim.get(&Semantic::TexCoords(0));
		let mut uv = uv
			.map(|uv| {
				let (uv, ty, comp) = self.accessor(uv)?;
				if comp != Dimensions::Vec2 {
					return Err(ImportError::InvalidGltf);
				}
				let mut reader = SliceReader::new(uv);

				if !matches!(ty, DataType::F32 | DataType::U8 | DataType::U16) {
					return Err(ImportError::InvalidGltf);
				}
				Ok(std::iter::from_fn(move || match ty {
					DataType::F32 => (!reader.is_empty()).then(|| reader.read::<Vec2<f32>>()),
					DataType::U8 => (!reader.is_empty()).then(|| reader.read::<Vec2<u8>>().map(|u| u as f32 / 255.0)),
					DataType::U16 => {
						(!reader.is_empty()).then(|| reader.read::<Vec2<u16>>().map(|u| u as f32 / 65535.0))
					},
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
		{
			let s = span!(Level::INFO, "optimizing mesh");
			let _e = s.enter();

			let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(&indices));
			let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap);
			let mut indices = meshopt::remap_index_buffer(Some(&indices), vertex_count, &remap);
			meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());
			meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);
		}

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
		let mut mesh = Mesh {
			vertices: Vec::with_capacity(vertices.len()),
			indices: Vec::with_capacity(indices.len()),
			meshlets: Vec::with_capacity(meshlets.len()),
			aabb: Aabb {
				min: Vec3::broadcast(f32::INFINITY),
				max: Vec3::broadcast(f32::NEG_INFINITY),
			},
		};
		for m in meshlets.iter() {
			let bounds = meshopt::compute_meshlet_bounds(m, &adapter);
			let vertices = m.vertices.iter().map(|&x| vertices[x as usize]);
			let aabb = {
				let mut min = Vec3::broadcast(f32::INFINITY);
				let mut max = Vec3::broadcast(f32::NEG_INFINITY);
				for v in vertices.clone() {
					min.x = min.x.min(v.position.x);
					min.y = min.y.min(v.position.y);
					min.z = min.z.min(v.position.z);
					max.x = max.x.max(v.position.x);
					max.y = max.y.max(v.position.y);
					max.z = max.z.max(v.position.z);
				}
				Aabb { min, max }
			};
			let extent = aabb.max - aabb.min;

			let index_offset = mesh.indices.len() as u32;
			let vertex_offset = mesh.vertices.len() as u32;
			let vert_count = m.vertices.len() as u8;
			let tri_count = (m.triangles.len() / 3) as u8;
			mesh.vertices.extend(vertices.map(|x| Vertex {
				position: ((x.position - aabb.min) / extent * Vec3::broadcast(65535.0)).map(|x| x.round() as u16),
				normal: (x.normal * Vec3::broadcast(32767.0)).map(|x| x.round() as i16),
				uv: (x.uv * Vec2::broadcast(65535.0)).map(|x| x.round() as u16),
			}));
			mesh.indices.extend(m.triangles.iter().map(|&x| x as u32));
			mesh.meshlets.push(Meshlet {
				aabb_min: aabb.min,
				aabb_extent: extent,
				index_offset,
				vertex_offset,
				tri_count,
				vert_count,
				cone: Cone {
					axis: Vec3::from_slice(&bounds.cone_axis_s8),
					cutoff: bounds.cone_cutoff_s8,
				},
				_pad: 0,
			});
			mesh.aabb = mesh.aabb.union(aabb);
		}

		Ok(mesh)
	}

	fn scene(&self, name: &str, scene: gltf::Scene, models: &[Uuid]) -> ImportResult<Scene> {
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
			self.node(node, Mat4::identity(), models, &mut out);
		}

		Ok(out)
	}

	fn node(&self, node: gltf::Node, transform: Mat4<f32>, models: &[Uuid], out: &mut Scene) {
		let this_transform = Mat4::from_col_arrays(node.transform().matrix());
		let transform = transform * this_transform;

		let model = node.mesh().map(|model| {
			let name = node.name().unwrap_or("unnamed node").to_string();
			let model = models[model.index()];

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
			self.node(child, transform, models, out);
		}
	}

	pub fn accessor(&self, accessor: Accessor) -> ImportResult<(&[u8], DataType, Dimensions)> {
		if accessor.sparse().is_some() {
			return Err(ImportError::UnsupportedFeature);
		}

		let view = accessor.view().ok_or(ImportError::InvalidGltf)?;
		let buffer = &self.buffers[view.buffer().index()];
		let offset = accessor.offset() + view.offset();
		let data = &buffer[offset..offset + view.length()];
		Ok((data, accessor.data_type(), accessor.dimensions()))
	}
}
