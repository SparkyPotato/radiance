use std::{
	fs::{self, File},
	io::{self, BufReader},
	path::{Path, PathBuf},
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc,
	},
	time::SystemTime,
};

use gltf::{
	buffer,
	camera::Projection,
	image::{self, Source},
	Document,
	Gltf,
};
use rad_core::{
	asset::{aref::ARef, Asset, AssetId},
	Engine,
};
use rad_renderer::{
	assets::{
		image::{Image, ImportImage},
		material::Material,
		mesh::{GpuVertex, Mesh, MeshData},
	},
	components::{
		camera::CameraComponent,
		light::{LightComponent, LightType},
		mesh::MeshComponent,
	},
	vek::{Mat4, Quaternion, Vec2, Vec3},
};
use rad_world::{transform::Transform, World};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use tracing::{info, span, trace_span, Level};

use crate::asset::fs::FsAssetSystem;

pub struct GltfImporter {
	gltf: Document,
	base: PathBuf,
	buffers: Vec<buffer::Data>,
}

#[derive(Copy, Clone)]
struct ImportProgress {
	images: u32,
	materials: u32,
	meshes: u32,
	scenes: u32,
}

impl ImportProgress {
	fn ratio(&self, total: ImportProgress) -> f32 {
		(self.images + self.materials + self.meshes + self.scenes) as f32
			/ (total.images + total.materials + total.meshes + total.scenes) as f32
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
			images: self.gltf.images().count() as _,
			materials: self.gltf.materials().count() as _,
			meshes: self.gltf.meshes().count() as _,
			scenes: self.gltf.scenes().count() as _,
		};
		progress(0.0);
		let sys: &Arc<FsAssetSystem> = Engine::get().asset_source().unwrap();

		let prog = AtomicUsize::new(0);
		let images: Vec<_> = self.gltf.images().collect();
		let images: Vec<_> = images
			.into_par_iter()
			.map(|image| {
				let meta_path = match image.source() {
					Source::Uri { uri, .. } => fs::canonicalize(self.base.join(uri)).ok(),
					_ => None,
				};
				if let Some(ref p) = meta_path {
					if let Some((id, modified)) = sys.import_source_of(&p) {
						if modified
							>= p.metadata()
								.ok()
								.and_then(|x| x.modified().ok())
								.unwrap_or(SystemTime::UNIX_EPOCH)
						{
							info!("skipping image {}", p.display());
							return Ok(ARef::unloaded(id));
						}
					}
				}

				let id = AssetId::new();
				let name = image
					.name()
					.map(|x| x.to_string())
					.or_else(|| {
						let Source::Uri { uri, .. } = image.source() else {
							return None;
						};
						Some(uri.to_string())
					})
					.unwrap_or_else(|| id.to_string());
				let path = Path::new("images").join(&name);
				let data = {
					let s = trace_span!("load image", name = name);
					let _e = s.enter();
					image::Data::from_source(image.source(), Some(self.base.as_path()), &self.buffers)
						.map_err(io::Error::other)?
				};
				Image::import(
					&name,
					ImportImage {
						data: &data.pixels,
						width: data.width,
						height: data.height,
						// TODO: yeah
						is_normal_map: false,
						is_srgb: true,
					},
					sys.create(&path, id, Image::uuid(), meta_path)?,
				)?;
				let old = prog.fetch_add(1, Ordering::Relaxed);
				progress(
					ImportProgress {
						images: old as u32 + 1,
						materials: 0,
						meshes: 0,
						scenes: 0,
					}
					.ratio(total),
				);

				Ok::<_, io::Error>(ARef::unloaded(id))
			})
			.collect::<Result<_, _>>()?;

		let prog = AtomicUsize::new(0);
		let materials: Vec<_> = self.gltf.materials().collect();
		let materials: Vec<_> = materials
			.into_par_iter()
			.map(|mat| {
				let id = AssetId::new();
				let name = mat.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string());
				let path = Path::new("materials").join(&name);
				let mat = self.material(&name, mat, &images);
				mat.save(&mut sys.create(&path, id, Material::uuid(), None)?)?;
				let old = prog.fetch_add(1, Ordering::Relaxed);
				progress(
					ImportProgress {
						images: total.images,
						materials: old as u32 + 1,
						meshes: 0,
						scenes: 0,
					}
					.ratio(total),
				);

				Ok::<_, io::Error>(ARef::unloaded(id))
			})
			.collect::<Result<_, _>>()?;

		let prog = AtomicUsize::new(0);
		let meshes: Vec<_> = self.gltf.meshes().collect();
		let meshes: Vec<_> = meshes
			.into_par_iter()
			.map(|mesh| {
				let name = mesh.name().map(|x| x.to_string());
				let prims = self.conv_to_meshes(mesh, &materials).map_err(io::Error::other)?;
				let c = prims.len();
				let ids = prims
					.into_iter()
					.enumerate()
					.map(|(i, m)| {
						let id = AssetId::new();
						let name = name.clone().unwrap_or_else(|| id.to_string());
						let name = if c == 1 {
							name.to_string()
						} else {
							format!("{name}-{i}")
						};
						let path = Path::new("meshes").join(&name);
						Mesh::import(&name, m, sys.create(&path, id, Mesh::uuid(), None)?)?;
						Ok::<_, io::Error>(ARef::unloaded(id))
					})
					.collect::<Result<Vec<_>, _>>()?;

				let old = prog.fetch_add(1, Ordering::Relaxed);
				progress(
					ImportProgress {
						images: total.images,
						materials: total.materials,
						meshes: old as u32 + 1,
						scenes: 0,
					}
					.ratio(total),
				);

				Ok(ids)
			})
			.collect::<Result<_, io::Error>>()?;

		let prog = AtomicUsize::new(0);
		self.gltf.scenes().par_bridge().try_for_each(|scene| {
			let id = AssetId::new();
			let name = scene.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string());
			let path = Path::new("scenes").join(&name);
			let world = self.scene(&name, scene, &meshes).map_err(io::Error::other)?;
			world.save(sys.create(&path, id, World::uuid(), None)?.as_mut())?;
			let old = prog.fetch_add(1, Ordering::Relaxed);
			progress(
				ImportProgress {
					images: total.images,
					materials: total.materials,
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
		Ok(Self {
			gltf,
			base: base.to_path_buf(),
			buffers,
		})
	}

	fn material(&self, name: &str, mat: gltf::Material, images: &[ARef<Image>]) -> Material {
		let s = span!(Level::INFO, "importing material", name = name);
		let _e = s.enter();

		let m = mat.pbr_metallic_roughness();

		Material {
			base_color: m
				.base_color_texture()
				.map(|x| images[x.texture().source().index()].clone()),
			base_color_factor: m.base_color_factor().into(),
			metallic_roughness: m
				.metallic_roughness_texture()
				.map(|x| images[x.texture().source().index()].clone()),
			metallic_factor: m.metallic_factor(),
			roughness_factor: m.roughness_factor(),
			normal: mat
				.normal_texture()
				.map(|x| images[x.texture().source().index()].clone()),
			emissive: mat
				.emissive_texture()
				.map(|x| images[x.texture().source().index()].clone()),
			emissive_factor: mat.emissive_factor().into(),
		}
	}

	fn scene(&self, name: &str, scene: gltf::Scene, meshes: &[Vec<ARef<Mesh>>]) -> Result<World, gltf::Error> {
		let s = span!(Level::INFO, "importing scene", name = name);
		let _e = s.enter();

		let mut out = World::new();
		for node in scene.nodes() {
			self.node(node, Mat4::identity(), meshes, &mut out);
		}

		Ok(out)
	}

	fn node(&self, node: gltf::Node, transform: Mat4<f32>, meshes: &[Vec<ARef<Mesh>>], out: &mut World) {
		// let name = node.name().unwrap_or("unnamed node").to_string();

		let this_transform = Mat4::from_col_arrays(node.transform().matrix());
		let transform = transform * this_transform;

		let mut entity = out.spawn_empty();

		// gltf is X- right, Y up, Z in
		// we are X right, Y in, Z up
		let basis_change = Mat4::new(
			1.0, 0.0, 0.0, 0.0, //
			0.0, 0.0, -1.0, 0.0, //
			0.0, 1.0, 0.0, 0.0, //
			0.0, 0.0, 0.0, 1.0, //
		);
		let (p, r, s) = gltf::scene::Transform::Matrix {
			matrix: (basis_change * transform).into_col_arrays(),
		}
		.decomposed();
		entity.insert(Transform {
			position: p.into(),
			rotation: Quaternion::from_vec4(r.into()),
			scale: s.into(),
		});

		if let Some(mesh) = node.mesh() {
			entity.insert(MeshComponent::new(&meshes[mesh.index()].clone()));
		}

		if let Some(light) = node.light() {
			entity.insert(LightComponent {
				ty: match light.kind() {
					gltf::khr_lights_punctual::Kind::Directional => LightType::Directional,
					gltf::khr_lights_punctual::Kind::Point => LightType::Point,
					_ => LightType::Directional,
				},
				radiance: Vec3::from(light.color()) * light.intensity(),
			});
		}

		if let Some(Projection::Perspective(p)) = node.camera().as_ref().map(|x| x.projection()) {
			entity.insert(CameraComponent {
				fov: p.yfov(),
				near: p.znear(),
			});
		}

		for child in node.children() {
			self.node(child, transform, meshes, out);
		}
	}

	fn conv_to_meshes(&self, mesh: gltf::Mesh, materials: &[ARef<Material>]) -> Result<Vec<MeshData>, io::Error> {
		let s = trace_span!("convert from gltf");
		let _e = s.enter();

		let out = mesh
			.primitives()
			.map(|prim| {
				let reader = prim.reader(|x| Some(&self.buffers[x.index()]));
				let positions = reader
					.read_positions()
					.ok_or_else(|| io::Error::other("invalid gltf"))?
					.map(|x| x.into());
				let normals = reader
					.read_normals()
					.ok_or_else(|| io::Error::other("invalid gltf"))?
					.map(|x| x.into());
				let mut uvs = reader.read_tex_coords(0).map(|x| x.into_f32());

				let indices = reader
					.read_indices()
					.ok_or_else(|| io::Error::other("invalid gltf"))?
					.into_u32()
					.collect();

				let vertices = positions
					.zip(normals)
					.zip(std::iter::from_fn(move || {
						if let Some(ref mut uvs) = uvs {
							uvs.next().map(Into::into)
						} else {
							Some(Vec2::new(0.0, 0.0))
						}
					}))
					.map(|((position, normal), uv)| GpuVertex { position, normal, uv })
					.collect();

				Ok::<_, io::Error>(MeshData {
					vertices,
					indices,
					material: materials[prim.material().index().ok_or_else(|| {
						io::Error::new(io::ErrorKind::Unsupported, "gltf default material unsupported")
					})?]
					.clone(),
				})
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(out)
	}
}
