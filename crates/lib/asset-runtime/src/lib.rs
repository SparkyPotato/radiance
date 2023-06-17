#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.

use std::ops::Range;

use ash::vk;
use bytemuck::{cast_slice, NoUninit};
pub use radiance_asset::mesh::{Cone, Vertex};
use radiance_asset::{mesh, scene, Asset, AssetSource, AssetSystem};
use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::resource::BufferDesc;
use radiance_util::{buffer::StretchyBuffer, staging::StageTicket};
use rustc_hash::FxHashMap;
use static_assertions::const_assert_eq;
use tracing::span;
use uuid::Uuid;
use vek::{Mat4, Vec3, Vec4};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Meshlet {
	/// The transformation of the meshlet in world space. The AABB is a cube from `[0, 1]`.
	pub transform: Vec4<Vec3<f32>>,
	/// Start index of the meshlet in the global index buffer.
	pub start_index: u32,
	/// Start vertex of the meshlet in the global vertex buffer.
	pub start_vertex: u32,
	/// Cone of the meshlet relative to the center of the bounding box.
	pub cone: Cone,
	/// Number of triangles in the meshlet. The number of indices will be 3 times this.
	pub tri_count: u8,
	/// Number of vertices in the meshlet.
	pub vert_count: u8,
	pub _pad: u16,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Camera {
	pub view: Vec4<Vec3<f32>>,
	pub projection: Vec4<Vec3<f32>>,
}

const_assert_eq!(std::mem::size_of::<Meshlet>(), 64);
const_assert_eq!(std::mem::align_of::<Meshlet>(), 4);

pub struct Scene {
	pub meshlets: StretchyBuffer,
	pub vertices: StretchyBuffer,
	pub indices: StretchyBuffer,
	pub cameras: Vec<scene::Camera>,
}

struct Model {
	meshlets: Vec<mesh::Meshlet>,
}

pub struct AssetRuntime {
	scenes: FxHashMap<Uuid, Scene>,
}

impl AssetRuntime {
	pub const INDEX_SIZE: u64 = std::mem::size_of::<u16>() as u64;
	pub const MESHLET_SIZE: u64 = std::mem::size_of::<Meshlet>() as u64;
	pub const START_INDEX_COUNT: u64 = Self::START_MESHLET_COUNT * 64 * 3;
	pub const START_MESHLET_COUNT: u64 = 8192;
	pub const START_VERTEX_COUNT: u64 = Self::START_MESHLET_COUNT * 124;
	pub const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

	pub fn new() -> Self {
		Self {
			scenes: FxHashMap::default(),
		}
	}

	pub fn unload_scene(&mut self, core: &mut RenderCore, asset: Uuid) {
		if let Some(scene) = self.scenes.remove(&asset) {
			unsafe {
				scene.vertices.delete(&mut core.delete);
				scene.indices.delete(&mut core.delete);
				scene.meshlets.delete(&mut core.delete);
			}
		}
	}

	pub fn get_scene(&self, asset: Uuid) -> Option<&Scene> { self.scenes.get(&asset) }

	pub fn load_scene<S: AssetSource>(
		&mut self, device: &CoreDevice, core: &mut RenderCore, scene: Uuid, system: &mut AssetSystem<S>,
	) -> Result<Option<StageTicket>, S::Error> {
		let s = span!(tracing::Level::INFO, "load_scene", scene = %scene);
		let _e = s.enter();

		if self.scenes.contains_key(&scene) {
			return Ok(None);
		}

		let uuid = scene;
		let scene = match system.load(scene)? {
			Asset::Scene(scene) => scene,
			_ => unreachable!("Scene asset is not a scene"),
		};

		let mut out = Scene {
			meshlets: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_MESHLET_COUNT * Self::MESHLET_SIZE,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				},
			)
			.unwrap(),
			indices: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_INDEX_COUNT * Self::INDEX_SIZE,
					usage: vk::BufferUsageFlags::INDEX_BUFFER,
				},
			)
			.unwrap(),
			vertices: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_VERTEX_COUNT * Self::VERTEX_SIZE,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				},
			)
			.unwrap(),
			cameras: scene.cameras,
		};

		let mut model_map = FxHashMap::default();
		let mut meshlets = Vec::new();
		let mut vertices = Vec::new();
		let mut indices = Vec::new();
		for node in scene.nodes.iter() {
			let model = if let Some(m) = model_map.get(&node.model) {
				m
			} else {
				let model = match system.load(node.model)? {
					Asset::Model(m) => m,
					_ => unreachable!("Model asset is not a model"),
				};
				let mut meshlets = Vec::new();
				for mesh in model.meshes {
					let m = match system.load(mesh)? {
						Asset::Mesh(m) => m,
						_ => unreachable!("Mesh asset is not a mesh"),
					};

					let vertex_offset = vertices.len() as u32;
					let index_offset = indices.len() as u32;
					vertices.extend(m.vertices);
					indices.extend(m.indices.into_iter().map(|x| x as u8));
					meshlets.extend(m.meshlets.into_iter().map(|mut m| {
						m.vertex_offset += vertex_offset;
						m.index_offset += index_offset;
						m
					}));
				}

				model_map.insert(node.model, Model { meshlets });
				model_map.get(&node.model).unwrap()
			};

			meshlets.extend(model.meshlets.iter().map(|x| {
				let extent = x.aabb_max - x.aabb_min;
				let scale = Mat4::scaling_3d(extent);
				let translate = Mat4::translation_3d(x.aabb_min);
				let transform = node.transform * translate * scale;

				Meshlet {
					transform: transform.cols.map(|x| x.xyz()),
					start_index: x.index_offset,
					start_vertex: x.vertex_offset,
					tri_count: x.tri_count,
					vert_count: x.vert_count,
					cone: x.cone,
					_pad: 0,
				}
			}));
		}

		let ticket = core
			.stage(device, |ctx, delete| {
				out.vertices.push(ctx, delete, cast_slice(&vertices))?;
				out.indices.push(ctx, delete, cast_slice(&indices))?;
				out.meshlets.push(ctx, delete, cast_slice(&meshlets))?;

				Ok(())
			})
			.unwrap();

		self.scenes.insert(uuid, out);

		Ok(Some(ticket))
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		for (_, scene) in self.scenes {
			unsafe {
				scene.vertices.destroy(device);
				scene.indices.destroy(device);
				scene.meshlets.destroy(device);
			}
		}
	}
}
