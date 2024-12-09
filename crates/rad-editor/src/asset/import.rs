use std::{
	fs::File,
	io::{self, BufReader},
	path::Path,
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc,
	},
};

use bytemuck::from_bytes;
use gltf::{
	accessor::{DataType, Dimensions},
	buffer,
	Document,
	Gltf,
};
use rad_core::{
	asset::{aref::ARef, Asset, AssetId},
	Engine,
};
use rad_renderer::{
	assets::mesh::{GpuVertex, Mesh, MeshData},
	components::mesh::MeshComponent,
	vek::{Mat4, Quaternion, Vec2, Vec3},
};
use rad_world::{transform::Transform, World};
use rayon::iter::{ParallelBridge, ParallelIterator};
use tracing::{span, trace_span, Level};

use crate::asset::fs::FsAssetSystem;

pub struct GltfImporter {
	gltf: Document,
	buffers: Vec<buffer::Data>,
}

#[derive(Copy, Clone)]
struct ImportProgress {
	meshes: u32,
	scenes: u32,
}

impl ImportProgress {
	fn ratio(&self, total: ImportProgress) -> f32 {
		(self.meshes + self.scenes) as f32 / (total.meshes + total.scenes) as f32
	}
}

impl GltfImporter {
	pub fn initialize(path: &Path) -> Option<Result<Self, io::Error>> {
		if path.extension().and_then(|x| x.to_str()) != Some("gltf") {
			return None;
		}

		let s = span!(Level::TRACE, "load GLTF");
		let _e = s.enter();
		let base = path.parent().unwrap_or_else(|| Path::new("."));
		let file = match File::open(path) {
			Ok(x) => x,
			Err(e) => return Some(Err(e)),
		};
		let Gltf { document: gltf, blob } = match Gltf::from_reader(BufReader::new(file)) {
			Ok(x) => x,
			Err(e) => return Some(Err(io::Error::other(e))),
		};

		Some(Self::new(base, gltf, blob).map_err(|e| io::Error::other(e)))
	}

	pub fn import(self, progress: impl Fn(f32) + Send + Sync) -> Result<(), io::Error> {
		let total = ImportProgress {
			meshes: self.gltf.meshes().count() as _,
			scenes: self.gltf.scenes().count() as _,
		};
		progress(0.0);
		let sys: &Arc<FsAssetSystem> = Engine::get().asset_source().unwrap();

		let prog = AtomicUsize::new(0);
		let meshes: Vec<_> = self
			.gltf
			.meshes()
			.par_bridge()
			.map(|mesh| {
				let id = AssetId::new();
				let name = mesh.name().unwrap_or("unnamed mesh");
				let path =
					Path::new("meshes").join(mesh.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string()));
				let mesh = self.conv_to_mesh(mesh).map_err(io::Error::other)?;
				Mesh::import(name, mesh, sys.create(&path, id, Mesh::uuid())?)?;
				let old = prog.fetch_add(1, Ordering::Relaxed);
				progress(
					ImportProgress {
						meshes: old as u32 + 1,
						scenes: 0,
					}
					.ratio(total),
				);

				Engine::get().asset::<Mesh>(id)
			})
			.collect::<Result<_, _>>()?;

		let prog = AtomicUsize::new(0);
		self.gltf.scenes().par_bridge().try_for_each(|scene| {
			let id = AssetId::new();
			let name = scene.name().unwrap_or("unnamed scene");
			let path = Path::new("scenes").join(scene.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string()));
			let world = self.scene(&name, scene, &meshes).map_err(io::Error::other)?;
			world.save(sys.create(&path, id, World::uuid())?.as_mut())?;
			let old = prog.fetch_add(1, Ordering::Relaxed);
			progress(
				ImportProgress {
					meshes: total.meshes,
					scenes: old as u32 + 1,
				}
				.ratio(total),
			);

			Ok(())
		})
	}

	fn new(base: &Path, gltf: Document, mut blob: Option<Vec<u8>>) -> Result<Self, gltf::Error> {
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
		Ok(Self { gltf, buffers })
	}

	fn scene(&self, name: &str, scene: gltf::Scene, meshes: &[ARef<Mesh>]) -> Result<World, gltf::Error> {
		let s = span!(Level::INFO, "importing scene", name = name);
		let _e = s.enter();

		let mut out = World::new();
		for node in scene.nodes() {
			self.node(node, Mat4::identity(), meshes, &mut out);
		}

		Ok(out)
	}

	fn node(&self, node: gltf::Node, transform: Mat4<f32>, meshes: &[ARef<Mesh>], out: &mut World) {
		let this_transform = Mat4::from_col_arrays(node.transform().matrix());
		let transform = transform * this_transform;

		if let Some(mesh) = node.mesh() {
			// let name = node.name().unwrap_or("unnamed node").to_string();
			let mesh = meshes[mesh.index()].clone();
			let (t, r, s) = gltf::scene::Transform::Matrix {
				matrix: transform.into_col_arrays(),
			}
			.decomposed();

			out.spawn_empty()
				.insert(Transform {
					position: t.into(),
					rotation: Quaternion::from_vec4(r.into()),
					scale: s.into(),
				})
				.insert(MeshComponent::new(mesh));
		}

		// let camera = node.camera().map(|camera| {
		// 	let name = camera.name().unwrap_or("unnamed camera").to_string();
		// 	let view = transform.inverted();
		// 	let projection = match camera.projection() {
		// 		Projection::Perspective(p) => {
		// 			let yfov = p.yfov();
		// 			let near = p.znear();
		// 			let far = p.zfar();
		// 			crate::scene::Projection::Perspective { yfov, near, far }
		// 		},
		// 		Projection::Orthographic(o) => {
		// 			let height = o.ymag();
		// 			let near = o.znear();
		// 			let far = o.zfar();
		// 			crate::scene::Projection::Orthographic { height, near, far }
		// 		},
		// 	};
		// 	Camera { name, view, projection }
		// });
		// out.cameras.extend(camera);

		for child in node.children() {
			self.node(child, transform, meshes, out);
		}
	}

	pub fn accessor(
		&self, accessor: gltf::Accessor,
	) -> Result<(impl Iterator<Item = &[u8]>, DataType, Dimensions), gltf::Error> {
		if accessor.sparse().is_some() {
			return Err(gltf::Error::Io(io::Error::other("unsupported feature")));
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

		let view = accessor
			.view()
			.ok_or(gltf::Error::Io(io::Error::other("invalid gltf")))?;
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

	fn conv_to_mesh(&self, mesh: gltf::Mesh) -> Result<MeshData, gltf::Error> {
		let s = trace_span!("convert from gltf");
		let _e = s.enter();

		let mut out = MeshData {
			vertices: Vec::with_capacity(
				mesh.primitives()
					.flat_map(|x| x.get(&gltf::Semantic::Positions).map(|x| x.count()))
					.sum(),
			),
			indices: Vec::with_capacity(mesh.primitives().flat_map(|x| x.indices().map(|x| x.count())).sum()),
		};
		for prim in mesh.primitives() {
			let positions = prim
				.get(&gltf::Semantic::Positions)
				.ok_or(gltf::Error::Io(io::Error::other("invalid gltf")))?;
			let (positions, ty, comp) = self.accessor(positions)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(gltf::Error::Io(io::Error::other("invalid gltf")));
			}
			let positions = positions.map(|p| *from_bytes::<Vec3<f32>>(p));

			let normals = prim
				.get(&gltf::Semantic::Normals)
				.ok_or_else(|| gltf::Error::Io(io::Error::other("invalid gltf")))?;
			let (normals, ty, comp) = self.accessor(normals)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(gltf::Error::Io(io::Error::other("invalid gltf")));
			}
			let normals = normals.map(|n| *from_bytes::<Vec3<f32>>(n));

			let uv = prim.get(&gltf::Semantic::TexCoords(0));
			let mut uv = uv
				.map(|uv| {
					let (uv, ty, comp) = self.accessor(uv)?;
					if comp != Dimensions::Vec2 {
						return Err(gltf::Error::Io(io::Error::other("invalid gltf")));
					}

					if !matches!(ty, DataType::F32 | DataType::U8 | DataType::U16) {
						return Err(gltf::Error::Io(io::Error::other("invalid gltf")));
					}
					Ok(uv.map(move |uv| match ty {
						DataType::F32 => *from_bytes(uv),
						DataType::U8 => from_bytes::<Vec2<u8>>(uv).map(|u| u as f32 / 255.0),
						DataType::U16 => from_bytes::<Vec2<u16>>(uv).map(|u| u as f32 / 65535.0),
						_ => panic!("yikes"),
					}))
				})
				.transpose()?;

			let indices = prim
				.indices()
				.ok_or_else(|| gltf::Error::Io(io::Error::other("invalid gltf")))?;
			let (indices, ty, comp) = self.accessor(indices)?;
			if comp != Dimensions::Scalar {
				return Err(gltf::Error::Io(io::Error::other("invalid gltf")));
			}
			let offset = out.vertices.len() as u32;
			match ty {
				DataType::U8 => out.indices.extend(indices.flatten().map(|&i| i as u32 + offset)),
				DataType::U16 => out
					.indices
					.extend(indices.map(|i| *from_bytes::<u16>(i) as u32 + offset)),
				DataType::U32 => out.indices.extend(indices.map(|i| *from_bytes::<u32>(i) + offset)),
				_ => return Err(gltf::Error::Io(io::Error::other("invalid gltf"))),
			}

			out.vertices.extend(
				positions
					.zip(normals)
					.zip(std::iter::from_fn(move || {
						if let Some(ref mut uv) = uv {
							uv.next()
						} else {
							Some(Vec2::new(0.0, 0.0))
						}
					}))
					.map(|((position, normal), uv)| GpuVertex { position, normal, uv }),
			);
		}

		Ok(out)
	}
}
