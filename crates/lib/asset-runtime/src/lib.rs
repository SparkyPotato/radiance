#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.

use ash::vk;
use bytemuck::{cast_slice, NoUninit};
pub use radiance_asset::mesh::{Cone, Vertex};
use radiance_asset::{scene, util::SliceWriter, Asset, AssetSource, AssetSystem};
use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	resource::{BufferDesc, GpuBuffer, Resource},
	Error,
};
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
	/// Mesh buffer containing meshlets + vertices + indices.
	pub mesh: BufferId,
	pub _pad: Vec3<u32>,
}

const_assert_eq!(std::mem::size_of::<Instance>(), 64);
const_assert_eq!(std::mem::align_of::<Instance>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Meshlet {
	/// The bounding box of the meshlet.
	pub aabb_min: Vec3<f32>,
	pub aabb_extent: Vec3<f32>,
	/// Cone of the meshlet relative to the center of the bounding box.
	pub cone: Cone,
	/// The offset into the mesh buffer of the vertices and indices.
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub tri_count: u8,
	_pad: Vec3<u16>,
}

const_assert_eq!(std::mem::size_of::<Meshlet>(), 48);
const_assert_eq!(std::mem::align_of::<Meshlet>(), 4);

pub struct Scene {
	pub instances: StretchyBuffer,
	pub meshlet_pointers: StretchyBuffer,
	pub cameras: Vec<scene::Camera>,
	pub models: Vec<Model>,
}
pub struct Mesh {
	pub buffer: GpuBuffer,
	pub meshlet_count: u32,
}

pub struct Model {
	pub meshes: Vec<Mesh>,
}

pub struct AssetRuntime {
	scenes: FxHashMap<Uuid, Scene>,
}

impl AssetRuntime {
	pub const INDEX_SIZE: u64 = std::mem::size_of::<u8>() as u64;
	pub const INSTANCE_SIZE: u64 = std::mem::size_of::<Instance>() as u64;
	pub const MESHLET_POINTER_SIZE: u64 = std::mem::size_of::<MeshletPointer>() as u64;
	pub const MESHLET_SIZE: u64 = std::mem::size_of::<Meshlet>() as u64;
	pub const START_MESHLET_COUNT: u64 = 4096;
	pub const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

	pub fn new() -> Self {
		Self {
			scenes: FxHashMap::default(),
		}
	}

	pub fn unload_scene(&mut self, core: &mut RenderCore, asset: Uuid) {
		if let Some(scene) = self.scenes.remove(&asset) {
			unsafe {
				scene.instances.delete(&mut core.delete);
				scene.meshlet_pointers.delete(&mut core.delete);
				for model in scene.models {
					for mesh in model.meshes {
						core.delete.delete(mesh.buffer);
					}
				}
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
			cameras: scene.cameras,
			models: Vec::new(),
		};

		let ticket = core
			.stage(device, |ctx, delete| {
				let mut model_map = FxHashMap::default();
				let mut instances = Vec::new();
				let mut meshlet_pointers = Vec::new();

				for node in scene.nodes.iter() {
					let model = if let Some(m) = model_map.get(&node.model) {
						m
					} else {
						let model = match system.load(node.model).unwrap() {
							Asset::Model(m) => m,
							_ => unreachable!("Model asset is not a model"),
						};
						let meshes = model
							.meshes
							.into_iter()
							.map(|mesh| {
								let m = match system.load(mesh).unwrap() {
									Asset::Mesh(m) => m,
									_ => unreachable!("Mesh asset is not a mesh"),
								};

								let meshlet_byte_len = m.meshlets.len() * Self::MESHLET_SIZE as usize;
								let vertex_byte_len = m.vertices.len() * Self::VERTEX_SIZE as usize;
								let index_byte_len = m.indices.len() / 3 * 4 * Self::INDEX_SIZE as usize;

								let mut data = vec![0u8; meshlet_byte_len + vertex_byte_len + index_byte_len];
								let mut writer = SliceWriter::new(&mut data);

								for m in m.meshlets.iter() {
									writer.write(Meshlet {
										aabb_min: m.aabb_min,
										aabb_extent: m.aabb_extent,
										cone: m.cone,
										vertex_byte_offset: meshlet_byte_len as u32
											+ m.vertex_offset * Self::VERTEX_SIZE as u32,
										index_byte_offset: meshlet_byte_len as u32
											+ vertex_byte_len as u32 + m.index_offset / 3
											* 4 * Self::INDEX_SIZE as u32,
										vertex_count: m.vert_count,
										tri_count: m.tri_count,
										_pad: Vec3::zero(),
									});
								}
								writer.write_slice(&m.vertices);
								for chunk in m.indices.chunks(3) {
									writer.write_slice(chunk);
									writer.write(0u8);
								}

								let buffer = GpuBuffer::create(
									device,
									BufferDesc {
										size: data.len() as _,
										usage: vk::BufferUsageFlags::STORAGE_BUFFER
											| vk::BufferUsageFlags::TRANSFER_DST,
									},
								)?;
								ctx.stage_buffer(&data, buffer.inner.inner(), 0)?;

								Ok::<_, Error>(Mesh {
									buffer,
									meshlet_count: m.meshlets.len() as u32,
								})
							})
							.collect::<Result<Vec<Mesh>, _>>()?;

						model_map.insert(node.model, Model { meshes });
						model_map.get(&node.model).unwrap()
					};

					for mesh in model.meshes.iter() {
						let instance = instances.len() as u32;
						instances.push(Instance {
							transform: node.transform.cols.map(|x| x.xyz()),
							mesh: mesh.buffer.inner.id().unwrap(),
							_pad: Default::default(),
						});
						meshlet_pointers
							.extend((0..mesh.meshlet_count).map(|meshlet| MeshletPointer { instance, meshlet }));
					}
				}

				out.instances.push(ctx, delete, cast_slice(&instances))?;
				out.meshlet_pointers.push(ctx, delete, cast_slice(&meshlet_pointers))?;
				out.models = model_map.into_values().collect();

				Ok(())
			})
			.unwrap();

		self.scenes.insert(uuid, out);

		Ok(Some(ticket))
	}

	pub unsafe fn destroy(self, device: &CoreDevice) {
		for (_, scene) in self.scenes {
			scene.instances.destroy(device);
			scene.meshlet_pointers.destroy(device);
			for model in scene.models {
				for mesh in model.meshes {
					mesh.buffer.destroy(device);
				}
			}
		}
	}
}
