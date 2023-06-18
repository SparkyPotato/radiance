#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.

use std::{iter::repeat, ops::Range};

use ash::vk;
use bytemuck::{cast_slice, NoUninit};
pub use radiance_asset::mesh::{Cone, Vertex};
use radiance_asset::{scene, Asset, AssetSource, AssetSystem};
use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::resource::BufferDesc;
use radiance_util::{buffer::StretchyBuffer, staging::StageTicket};
use rustc_hash::FxHashMap;
use static_assertions::const_assert_eq;
use tracing::span;
use uuid::Uuid;
use vek::{Vec3, Vec4};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct MeshletPointer {
	pub instance: u32,
	pub meshlet: u32,
}

const_assert_eq!(std::mem::size_of::<MeshletPointer>(), 8);
const_assert_eq!(std::mem::align_of::<MeshletPointer>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Instance {
	pub transform: Vec4<Vec3<f32>>,
	pub base_meshlet: u32,
}

const_assert_eq!(std::mem::size_of::<Instance>(), 52);
const_assert_eq!(std::mem::align_of::<Instance>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Meshlet {
	/// The oriented bounding box of the meshlet.
	pub aabb_min: Vec3<f32>,
	pub aabb_extent: Vec3<f32>,
	/// Cone of the meshlet relative to the center of the bounding box.
	pub cone: Cone,
}

const_assert_eq!(std::mem::size_of::<Meshlet>(), 28);
const_assert_eq!(std::mem::align_of::<Meshlet>(), 4);

pub struct Scene {
	pub meshlets: StretchyBuffer,
	pub instances: StretchyBuffer,
	pub meshlet_pointers: StretchyBuffer,
	pub vertices: StretchyBuffer,
	pub indices: StretchyBuffer,
	pub cameras: Vec<scene::Camera>,
}

struct Model {
	meshlets: Range<u32>,
}

pub struct AssetRuntime {
	scenes: FxHashMap<Uuid, Scene>,
}

impl AssetRuntime {
	pub const INDEX_SIZE: u64 = std::mem::size_of::<u8>() as u64;
	pub const INSTANCE_SIZE: u64 = std::mem::size_of::<Instance>() as u64;
	pub const MESHLET_POINTER_SIZE: u64 = std::mem::size_of::<MeshletPointer>() as u64;
	pub const MESHLET_SIZE: u64 = std::mem::size_of::<Meshlet>() as u64;
	pub const START_INDEX_COUNT: u64 = Self::START_MESHLET_COUNT * 124 * 3;
	pub const START_MESHLET_COUNT: u64 = 4096;
	pub const START_VERTEX_COUNT: u64 = Self::START_MESHLET_COUNT * 64;
	pub const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

	pub fn new() -> Self {
		Self {
			scenes: FxHashMap::default(),
		}
	}

	pub fn unload_scene(&mut self, core: &mut RenderCore, asset: Uuid) {
		if let Some(scene) = self.scenes.remove(&asset) {
			unsafe {
				scene.meshlets.delete(&mut core.delete);
				scene.instances.delete(&mut core.delete);
				scene.meshlet_pointers.delete(&mut core.delete);
				scene.vertices.delete(&mut core.delete);
				scene.indices.delete(&mut core.delete);
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
			instances: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_MESHLET_COUNT * Self::INSTANCE_SIZE,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				},
			)
			.unwrap(),
			meshlet_pointers: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_MESHLET_COUNT * Self::MESHLET_POINTER_SIZE * 8,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
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
			indices: StretchyBuffer::new(
				device,
				BufferDesc {
					size: Self::START_INDEX_COUNT * Self::INDEX_SIZE,
					usage: vk::BufferUsageFlags::INDEX_BUFFER,
				},
			)
			.unwrap(),
			cameras: scene.cameras,
		};

		let mut model_map = FxHashMap::default();
		let mut meshlets = Vec::new();
		let mut instances = Vec::new();
		let mut meshlet_pointers = Vec::new();
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
				let meshlet_start = meshlets.len() as u32;
				for mesh in model.meshes {
					let m = match system.load(mesh)? {
						Asset::Mesh(m) => m,
						_ => unreachable!("Mesh asset is not a mesh"),
					};

					meshlets.extend(m.meshlets.into_iter().map(|mesh| {
						let vertex_fill = 64 - mesh.vert_count;
						let tri_fill = 124 - mesh.tri_count;
						let index_fill = tri_fill as usize * 3;

						let start = mesh.vertex_offset as usize;
						let count = mesh.vert_count as usize;
						vertices.extend(
							m.vertices[start..start + count]
								.iter()
								.copied()
								.chain(repeat(Vertex::default()).take(vertex_fill as usize)),
						);

						let start = mesh.index_offset as usize;
						let count = mesh.tri_count as usize * 3;
						indices.extend(
							m.indices[start..start + count]
								.iter()
								.map(|&x| x as u8)
								.chain(repeat(0).take(index_fill)),
						);

						Meshlet {
							aabb_min: mesh.aabb_min,
							aabb_extent: mesh.aabb_extent,
							cone: mesh.cone,
						}
					}));
				}

				model_map.insert(
					node.model,
					Model {
						meshlets: meshlet_start..meshlets.len() as u32,
					},
				);
				model_map.get(&node.model).unwrap()
			};

			let instance = Instance {
				transform: node.transform.cols.map(|x| x.xyz()),
				base_meshlet: model.meshlets.start,
			};
			let instance_id = instances.len() as u32;
			instances.push(instance);
			meshlet_pointers.extend(model.meshlets.clone().map(|meshlet| MeshletPointer {
				instance: instance_id,
				meshlet: meshlet - instance.base_meshlet,
			}));
		}

		let ticket = core
			.stage(device, |ctx, delete| {
				out.meshlets.push(ctx, delete, cast_slice(&meshlets))?;
				out.instances.push(ctx, delete, cast_slice(&instances))?;
				out.meshlet_pointers.push(ctx, delete, cast_slice(&meshlet_pointers))?;
				out.vertices.push(ctx, delete, cast_slice(&vertices))?;
				out.indices.push(ctx, delete, cast_slice(&indices))?;

				Ok(())
			})
			.unwrap();

		self.scenes.insert(uuid, out);

		Ok(Some(ticket))
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		for (_, scene) in self.scenes {
			scene.meshlets.destroy(device);
			scene.instances.destroy(device);
			scene.meshlet_pointers.destroy(device);
			scene.vertices.destroy(device);
			scene.indices.destroy(device);
		}
	}
}
