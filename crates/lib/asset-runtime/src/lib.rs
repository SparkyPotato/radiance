#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.

use std::sync::Arc;

use ash::vk;
use bytemuck::{cast_slice, NoUninit};
pub use radiance_asset::mesh::Vertex;
use radiance_asset::{scene, util::SliceWriter, Asset, AssetError, AssetSource, AssetSystem};
use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::{
	device::descriptor::BufferId,
	resource::{BufferDesc, GpuBuffer, Resource},
};
use radiance_util::{
	buffer::StretchyBuffer,
	staging::{StageError, StageTicket, StagingCtx},
};
use rustc_hash::FxHashMap;
use static_assertions::const_assert_eq;
use tracing::span;
use uuid::Uuid;
use vek::{Aabb, Vec3, Vec4};

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
	/// Mesh buffer containing meshlets + meshlet data.
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
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	_pad: [u8; 2],
}

const_assert_eq!(std::mem::size_of::<Meshlet>(), 36);
const_assert_eq!(std::mem::align_of::<Meshlet>(), 4);

pub struct Model {
	pub meshes: Vec<Arc<Mesh>>,
	pub aabb: Aabb<f32>,
}

pub struct Scene {
	pub instances: StretchyBuffer,
	pub meshlet_pointers: StretchyBuffer,
	pub cameras: Vec<scene::Camera>,
	pub models: Vec<Arc<Model>>,
}

pub struct Mesh {
	pub buffer: GpuBuffer,
	pub meshlet_count: u32,
}

pub struct AssetRuntime {
	scenes: FxHashMap<Uuid, Arc<Scene>>,
	models: FxHashMap<Uuid, Arc<Model>>,
	meshes: FxHashMap<Uuid, Arc<Mesh>>,
}

impl AssetRuntime {
	const INSTANCE_SIZE: u64 = std::mem::size_of::<Instance>() as u64;
	const MESHLET_POINTER_SIZE: u64 = std::mem::size_of::<MeshletPointer>() as u64;
	const MESHLET_SIZE: u64 = std::mem::size_of::<Meshlet>() as u64;
	const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

	pub fn new() -> Self {
		Self {
			scenes: FxHashMap::default(),
			models: FxHashMap::default(),
			meshes: FxHashMap::default(),
		}
	}

	pub fn unload_scene(&mut self, core: &mut RenderCore, scene: Uuid) {
		if let Some(scene) = self.scenes.remove(&scene) {
			let scene = Arc::try_unwrap(scene)
				.ok()
				.expect("Cannot remove scene with references still alive");
			unsafe {
				scene.instances.delete(&mut core.delete);
				scene.meshlet_pointers.delete(&mut core.delete);
				for model in scene.models {
					if let Ok(model) = Arc::try_unwrap(model) {
						for mesh in model.meshes {
							if let Ok(mesh) = Arc::try_unwrap(mesh) {
								core.delete.delete(mesh.buffer);
							}
						}
					}
				}
			}
		}
	}

	pub fn get_scene(&self, scene: Uuid) -> Option<&Arc<Scene>> { self.scenes.get(&scene) }

	pub fn load_scene<S: AssetSource>(
		&mut self, device: &CoreDevice, core: &mut RenderCore, scene: Uuid, system: &AssetSystem<S>,
	) -> Result<(Arc<Scene>, Option<StageTicket>), StageError<AssetError<S>>> {
		if let Some(id) = self.scenes.get(&scene) {
			return Ok((id.clone(), None));
		}

		let s = span!(tracing::Level::INFO, "load_scene", scene = %scene);
		let _e = s.enter();

		let uuid = scene;
		let scene = match system.load(scene)? {
			Asset::Scene(scene) => scene,
			_ => unreachable!("Scene asset is not a scene"),
		};

		let (scene, ticket) = core
			.stage(device, |ctx, delete| {
				let mut instances = Vec::with_capacity(scene.nodes.len());
				let mut meshlet_pointers = Vec::with_capacity(scene.nodes.len() * 8);
				let mut models = Vec::with_capacity(scene.nodes.len());

				for node in scene.nodes.iter() {
					let model = self.load_model(device, ctx, system, node.model)?;

					for mesh in model.meshes.iter() {
						let instance = instances.len() as u32;
						instances.push(Instance {
							transform: node.transform.cols.map(|x| x.xyz()),
							mesh: mesh.buffer.inner.id().unwrap(),
							_pad: Vec3::zero(),
						});
						meshlet_pointers
							.extend((0..mesh.meshlet_count).map(|meshlet| MeshletPointer { instance, meshlet }));
					}

					models.push(model);
				}

				let mut scene = Scene {
					instances: StretchyBuffer::new(
						device,
						BufferDesc {
							size: instances.len() as u64 * Self::INSTANCE_SIZE,
							usage: vk::BufferUsageFlags::STORAGE_BUFFER,
						},
					)
					.unwrap(),
					meshlet_pointers: StretchyBuffer::new(
						device,
						BufferDesc {
							size: meshlet_pointers.len() as u64 * Self::MESHLET_POINTER_SIZE,
							usage: vk::BufferUsageFlags::STORAGE_BUFFER,
						},
					)
					.unwrap(),
					cameras: scene.cameras,
					models,
				};

				scene
					.instances
					.push(ctx, delete, cast_slice(&instances))
					.map_err(StageError::Vulkan)?;
				scene
					.meshlet_pointers
					.push(ctx, delete, cast_slice(&meshlet_pointers))
					.map_err(StageError::Vulkan)?;

				Ok::<_, StageError<_>>(Arc::new(scene))
			})
			.map_err(|x| match x {
				StageError::User(x) => match x {
					StageError::User(x) => StageError::User(x),
					StageError::Vulkan(x) => StageError::Vulkan(x),
				},
				StageError::Vulkan(x) => StageError::Vulkan(x),
			})?;

		self.scenes.insert(uuid, scene.clone());
		Ok((scene, Some(ticket)))
	}

	pub unsafe fn destroy(mut self, device: &CoreDevice) {
		self.models.clear();

		for (_, scene) in self.scenes {
			let scene = Arc::try_unwrap(scene)
				.ok()
				.expect("Cannot destroy `AssetRuntime` with asset references still alive");
			scene.instances.destroy(device);
			scene.meshlet_pointers.destroy(device);
		}

		for (_, mesh) in self.meshes {
			let mesh = Arc::try_unwrap(mesh)
				.ok()
				.expect("Cannot destroy `AssetRuntime` with asset references still alive");
			device.device().destroy_buffer(mesh.buffer.inner.inner(), None);
		}
	}

	fn load_model<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, system: &AssetSystem<S>, model: Uuid,
	) -> Result<Arc<Model>, StageError<AssetError<S>>> {
		if let Some(m) = self.models.get(&model) {
			Ok(m.clone())
		} else {
			let m = match system.load(model)? {
				Asset::Model(m) => m,
				_ => unreachable!("Model asset is not a model"),
			};
			let meshes = m
				.meshes
				.into_iter()
				.map(|mesh| self.load_mesh(device, ctx, system, mesh))
				.collect::<Result<Vec<_>, _>>()?;

			let m = Arc::new(Model { meshes, aabb: m.aabb });

			self.models.insert(model, m.clone());
			Ok(m)
		}
	}

	fn load_mesh<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, system: &AssetSystem<S>, mesh: Uuid,
	) -> Result<Arc<Mesh>, StageError<AssetError<S>>> {
		if let Some(m) = self.meshes.get(&mesh) {
			Ok(m.clone())
		} else {
			let m = match system.load(mesh)? {
				Asset::Mesh(m) => m,
				_ => unreachable!("Mesh asset is not a mesh"),
			};

			let meshlet_byte_len = m.meshlets.len() * Self::MESHLET_SIZE as usize;
			let vertex_byte_len = m.vertices.len() * Self::VERTEX_SIZE as usize;
			let index_byte_len = m.indices.len() / 3 * std::mem::size_of::<u32>();
			let index_offset = meshlet_byte_len + vertex_byte_len;

			let mut data = vec![0u8; index_offset + index_byte_len];
			let mut writer = SliceWriter::new(&mut data);

			for m in m.meshlets.iter() {
				writer
					.write(Meshlet {
						aabb_min: m.aabb_min,
						aabb_extent: m.aabb_extent,
						vertex_byte_offset: meshlet_byte_len as u32 + m.vertex_offset * Self::VERTEX_SIZE as u32,
						index_byte_offset: index_offset as u32 + m.index_offset / 3 * std::mem::size_of::<u32>() as u32,
						vertex_count: m.vert_count,
						triangle_count: m.tri_count,
						_pad: [0, 0],
					})
					.unwrap();
			}
			writer.write_slice(&m.vertices).unwrap();
			for tri in m.indices.chunks(3) {
				writer.write_slice(tri).unwrap();
				writer.write(0u8).unwrap();
			}

			let buffer = GpuBuffer::create(
				device,
				BufferDesc {
					size: data.len() as _,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				},
			)
			.map_err(StageError::Vulkan)?;
			ctx.stage_buffer(&data, buffer.inner.inner(), 0)
				.map_err(StageError::Vulkan)?;

			let m = Arc::new(Mesh {
				buffer,
				meshlet_count: m.meshlets.len() as u32,
			});
			self.meshes.insert(mesh, m.clone());
			Ok(m)
		}
	}
}
