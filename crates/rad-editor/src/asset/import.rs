use std::{
	collections::hash_map::Entry,
	fs::File,
	io::{self, BufReader},
	path::{Path, PathBuf},
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc,
	},
};

use gltf::{
	buffer,
	camera::Projection,
	image::{self, Source},
	Document,
	Gltf,
};
use parking_lot::Mutex;
use rad_core::{
	asset::{aref::AssetId, Asset},
	Engine,
};
use rad_graph::ash::vk;
use rad_renderer::{
	assets::{
		image::ImageAsset,
		material::Material,
		mesh::{GpuVertex, Mesh},
	},
	components::{
		camera::CameraComponent,
		light::{LightComponent, LightType},
		mesh::MeshComponent,
	},
	vek::{Mat4, Quaternion, Vec2, Vec3, Vec4},
};
use rad_world::{transform::Transform, World};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use rustc_hash::FxHashMap;
use tracing::{span, trace_span, Level};

use crate::asset::fs::FsAssetSystem;

pub struct GltfImporter {
	gltf: Document,
	base: PathBuf,
	buffers: Vec<buffer::Data>,
	image_cache: Mutex<FxHashMap<(usize, bool), AssetId<ImageAsset>>>,
}

#[derive(Copy, Clone)]
struct ImportProgress {
	materials: u32,
	meshes: u32,
	scenes: u32,
}

impl ImportProgress {
	fn ratio(&self, total: ImportProgress) -> f32 {
		(self.materials + self.meshes + self.scenes) as f32 / (total.materials + total.meshes + total.scenes) as f32
	}
}

impl GltfImporter {
	pub fn initialize(path: &Path) -> Option<Result<Self, io::Error>> {
		if path.extension().and_then(|x| x.to_str()) != Some("gltf") {
			return None;
		}

		let s = span!(Level::TRACE, "load gltf");
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
			materials: self.gltf.materials().count() as _,
			meshes: self.gltf.meshes().count() as _,
			scenes: self.gltf.scenes().count() as _,
		};
		progress(0.0);
		let sys: &Arc<FsAssetSystem> = Engine::get().asset_source();

		let prog = AtomicUsize::new(0);
		let materials: Vec<_> = self.gltf.materials().collect();
		let materials: Vec<_> = {
			let s = trace_span!("importing materials");
			let _e = s.enter();

			materials
				.into_par_iter()
				.map(|mat| {
					let id = AssetId::new();
					let name = mat.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string());
					let s = trace_span!("import material", name = name);
					let _e = s.enter();

					let path = Path::new("materials").join(&name);
					{
						let s = trace_span!("save");
						let _e = s.enter();
						let m = mat.pbr_metallic_roughness();
						let es = mat.emissive_strength().unwrap_or(1.0);
						Material {
							base_color: m
								.base_color_texture()
								.map(|x| self.image(x.texture().source(), true))
								.transpose()?,
							base_color_factor: m.base_color_factor().into(),
							metallic_roughness: m
								.metallic_roughness_texture()
								.map(|x| self.image(x.texture().source(), false))
								.transpose()?,
							metallic_factor: m.metallic_factor(),
							roughness_factor: m.roughness_factor(),
							normal: mat
								.normal_texture()
								.map(|x| self.image(x.texture().source(), false))
								.transpose()?,
							emissive: mat
								.emissive_texture()
								.map(|x| self.image(x.texture().source(), true))
								.transpose()?,
							emissive_factor: mat.emissive_factor().map(|x| x * es).into(),
						}
						.save(&mut sys.create(&path, id)?)?;
					}

					let old = prog.fetch_add(1, Ordering::Relaxed);
					progress(
						ImportProgress {
							materials: old as u32 + 1,
							meshes: 0,
							scenes: 0,
						}
						.ratio(total),
					);

					Ok::<_, io::Error>(id)
				})
				.chain(Some({
					let id = AssetId::new();
					let path = Path::new("materials").join("default");
					self.default_material().save(&mut sys.create(&path, id)?)?;
					Ok(id)
				}))
				.collect::<Result<_, _>>()?
		};

		let prog = AtomicUsize::new(0);
		let meshes: Vec<_> = self.gltf.meshes().collect();
		let meshes: Vec<_> = {
			let s = trace_span!("importing meshes");
			let _e = s.enter();

			meshes
				.into_par_iter()
				.map(|mesh| {
					let name = mesh.name().map(|x| x.to_string());
					let s = trace_span!("import mesh", name = name);
					let _e = s.enter();

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
							let s = trace_span!("save primitive", i = i);
							let _e = s.enter();

							let path = Path::new("meshes").join(&name);
							m.save(&mut sys.create(&path, id)?)?;
							Ok::<_, io::Error>(id)
						})
						.collect::<Result<Vec<_>, _>>()?;

					let old = prog.fetch_add(1, Ordering::Relaxed);
					progress(
						ImportProgress {
							materials: total.materials,
							meshes: old as u32 + 1,
							scenes: 0,
						}
						.ratio(total),
					);

					Ok(ids)
				})
				.collect::<Result<_, io::Error>>()?
		};

		let prog = AtomicUsize::new(0);
		{
			let s = trace_span!("importing scenes");
			let _e = s.enter();

			self.gltf.scenes().par_bridge().try_for_each(|scene| {
				let id = AssetId::<World>::new();
				let name = scene.name().map(|x| x.to_string()).unwrap_or_else(|| id.to_string());
				let s = trace_span!("import scene", name = name);
				let _e = s.enter();

				let path = Path::new("scenes").join(&name);
				let scene = self.scene(&name, scene, &meshes).map_err(io::Error::other)?;
				{
					let s = trace_span!("save");
					let _e = s.enter();
					scene.save(&mut sys.create(&path, id)?)?;
				}

				let old = prog.fetch_add(1, Ordering::Relaxed);
				progress(
					ImportProgress {
						materials: total.materials,
						meshes: total.meshes,
						scenes: old as u32 + 1,
					}
					.ratio(total),
				);

				Ok(())
			})
		}
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
			image_cache: Mutex::new(FxHashMap::default()),
		})
	}

	fn scene(&self, name: &str, scene: gltf::Scene, meshes: &[Vec<AssetId<Mesh>>]) -> Result<World, gltf::Error> {
		let s = span!(Level::INFO, "importing scene", name = name);
		let _e = s.enter();

		let mut out = World::new();
		for node in scene.nodes() {
			self.node(node, Mat4::identity(), meshes, &mut out);
		}

		Ok(out)
	}

	fn node(&self, node: gltf::Node, transform: Mat4<f32>, meshes: &[Vec<AssetId<Mesh>>], out: &mut World) {
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

	fn image(&self, image: gltf::Image, srgb: bool) -> Result<AssetId<ImageAsset>, io::Error> {
		let mut cache = self.image_cache.lock();
		let id = match cache.entry((image.index(), srgb)) {
			Entry::Occupied(x) => return Ok(*x.get()),
			Entry::Vacant(x) => *x.insert(AssetId::new()),
		};
		drop(cache);

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
		let s = trace_span!("import image", name = name);
		let _e = s.enter();

		let path = Path::new("images").join(&name);
		let mut d = {
			let s = trace_span!("load");
			let _e = s.enter();
			image::Data::from_source(image.source(), Some(self.base.as_path()), &self.buffers)
				.map_err(io::Error::other)?
		};
		if d.format == image::Format::R8G8B8 {
			let s = trace_span!("add alpha");
			let _e = s.enter();
			d.pixels = d
				.pixels
				.chunks_exact(3)
				.flat_map(|x| x.iter().copied().chain([255]))
				.collect();
			d.format = image::Format::R8G8B8A8;
		} else if d.format == image::Format::R8G8 && srgb {
			let s = trace_span!("add blue and alpha");
			let _e = s.enter();
			d.pixels = d
				.pixels
				.chunks_exact(2)
				.flat_map(|x| x.iter().copied().chain([0, 255]))
				.collect();
			d.format = image::Format::R8G8B8A8;
		}

		{
			let sys: &Arc<FsAssetSystem> = Engine::get().asset_source();
			let s = trace_span!("save");
			let _e = s.enter();
			ImageAsset {
				size: Vec3::new(d.width, d.height, 1),
				format: match (d.format, srgb) {
					(image::Format::R8, false) => vk::Format::R8_UNORM,
					(image::Format::R8G8, false) => vk::Format::R8G8_UNORM,
					(image::Format::R8G8B8A8, false) => vk::Format::R8G8B8A8_UNORM,
					(image::Format::R8, true) => vk::Format::R8_SRGB,
					(image::Format::R8G8B8A8, true) => vk::Format::R8G8B8A8_SRGB,
					(image::Format::R16, _) => vk::Format::R16_UNORM,
					(image::Format::R16G16, _) => vk::Format::R16G16_UNORM,
					(image::Format::R16G16B16, _) => vk::Format::R16G16B16_UNORM,
					(image::Format::R16G16B16A16, _) => vk::Format::R16G16B16A16_UNORM,
					(image::Format::R32G32B32FLOAT, _) => vk::Format::R32G32B32_SFLOAT,
					(image::Format::R32G32B32A32FLOAT, _) => vk::Format::R32G32B32A32_SFLOAT,
					_ => return Err(io::Error::other("unsupported image format")),
				}
				.as_raw(),
				data: d.pixels,
			}
			.save(&mut sys.create(&path, id)?)?;
		}

		Ok::<_, io::Error>(id)
	}

	fn default_material(&self) -> Material {
		Material {
			base_color: None,
			base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
			metallic_roughness: None,
			metallic_factor: 1.0,
			roughness_factor: 1.0,
			normal: None,
			emissive: None,
			emissive_factor: Vec3::zero(),
		}
	}

	fn conv_to_meshes(&self, mesh: gltf::Mesh, materials: &[AssetId<Material>]) -> Result<Vec<Mesh>, io::Error> {
		let s = trace_span!("load mesh");
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

				Ok::<_, io::Error>(Mesh {
					vertices,
					indices,
					material: materials[prim.material().index().unwrap_or(materials.len() - 1)].clone(),
				})
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(out)
	}
}
