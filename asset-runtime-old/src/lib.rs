#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.
//!
//! TODO: This entire crate is absolutely horrible and needs a rework.

use std::sync::Arc;

use ash::vk;
use bytemuck::{bytes_of, cast_slice, NoUninit};
pub use radiance_asset::mesh::Vertex;
use radiance_asset::{image::Format, scene, util::SliceWriter, Asset, AssetError, AssetSource, AssetSystem};
use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::{
	device::{
		descriptor::{BufferId, ImageId},
		QueueType,
	},
	resource::{
		ASDesc,
		BufferDesc,
		GpuBuffer,
		Image,
		ImageDesc,
		ImageView,
		ImageViewDesc,
		ImageViewUsage,
		Resource,
		AS,
	},
	sync::{ImageUsage, Shader},
};
use radiance_util::{
	buffer::StretchyBuffer,
	deletion::DeletionQueue,
	staging::{ImageStage, StageError, StageTicket, StagingCtx},
};
use rustc_hash::FxHashMap;
use static_assertions::const_assert_eq;
use tracing::span;
use uuid::Uuid;
use vek::{Aabb, Vec3, Vec4};

pub struct Model {
	pub meshes: Vec<Arc<Mesh>>,
	pub aabb: Aabb<f32>,
	pub blas: AS,
}

pub struct Scene {
	pub instances: StretchyBuffer,
	pub meshlet_pointers: StretchyBuffer,
	pub cameras: Vec<scene::Camera>,
	pub models: Vec<Arc<Model>>,
	pub tlas: AS,
}

impl Scene {
	pub fn meshlet_count(&self) -> u32 {
		(self.meshlet_pointers.len() / std::mem::size_of::<MeshletPointer>() as u64) as u32
	}
}

pub struct Mesh {
	pub buffer: GpuBuffer,
	pub meshlet_count: u32,
	pub material: u32,
	pub vertex_count: u32,
	pub tri_count: u32,
}

pub struct AssetRuntime {
	scenes: FxHashMap<Uuid, Arc<Scene>>,
	models: FxHashMap<Uuid, Arc<Model>>,
	meshes: FxHashMap<Uuid, Arc<Mesh>>,
	materials: FxHashMap<Uuid, u32>,
	images: FxHashMap<Uuid, Arc<ImageData>>,
	material_buf: StretchyBuffer,
}

impl AssetRuntime {
	const INSTANCE_SIZE: u64 = std::mem::size_of::<Instance>() as u64;
	const MESHLET_POINTER_SIZE: u64 = std::mem::size_of::<MeshletPointer>() as u64;
	const MESHLET_SIZE: u64 = std::mem::size_of::<Meshlet>() as u64;
	const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

	pub fn new(device: &CoreDevice) -> radiance_graph::Result<Self> {
		Ok(Self {
			scenes: FxHashMap::default(),
			models: FxHashMap::default(),
			meshes: FxHashMap::default(),
			materials: FxHashMap::default(),
			images: FxHashMap::default(),
			material_buf: StretchyBuffer::new(
				device,
				BufferDesc {
					size: std::mem::size_of::<Material>() as u64 * 1000,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				},
			)?,
		})
	}

	pub fn unload_scene(&mut self, core: &mut RenderCore, scene: Uuid) {
		if let Some(scene) = self.scenes.remove(&scene) {
			let scene = Arc::try_unwrap(scene)
				.ok()
				.expect("Cannot remove scene with references still alive");
			unsafe {
				scene.instances.delete(&mut core.delete);
				scene.meshlet_pointers.delete(&mut core.delete);
				core.delete.delete(scene.tlas);
				for model in scene.models {
					if let Ok(model) = Arc::try_unwrap(model) {
						core.delete.delete(model.blas);
						for mesh in model.meshes {
							if let Ok(mesh) = Arc::try_unwrap(mesh) {
								core.delete.delete(mesh.buffer);
							}
						}
					}
				}

				// TODO: Clear materials and unload images.
			}
		}
	}

	pub fn get_scene(&self, scene: Uuid) -> Option<&Arc<Scene>> { self.scenes.get(&scene) }

	pub fn load_scene<S: AssetSource>(
		&mut self, device: &CoreDevice, core: &mut RenderCore, system: &AssetSystem<S>, scene: Uuid,
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

		let ((scene, build_size), ticket) = core
			.stage(device, |ctx, delete| {
				let mut instances = Vec::with_capacity(scene.nodes.len());
				let mut meshlet_pointers = Vec::with_capacity(scene.nodes.len() * 8);
				let mut models = Vec::with_capacity(scene.nodes.len());

				let ty = vk::AccelerationStructureTypeKHR::TOP_LEVEL;
				let (size, build_size) = unsafe {
					let s = device.as_ext().get_acceleration_structure_build_sizes(
						vk::AccelerationStructureBuildTypeKHR::DEVICE,
						&vk::AccelerationStructureBuildGeometryInfoKHR::builder()
							.ty(ty)
							.flags(
								vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
									| vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
							)
							.geometries(&[vk::AccelerationStructureGeometryKHR::builder()
								.flags(vk::GeometryFlagsKHR::empty())
								.geometry_type(vk::GeometryTypeKHR::INSTANCES)
								.geometry(vk::AccelerationStructureGeometryDataKHR {
									instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder().build(),
								})
								.build()]),
						&[scene.nodes.len() as _],
					);
					(s.acceleration_structure_size, s.build_scratch_size)
				};
				let tlas = AS::create(
					device,
					ASDesc {
						ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
						flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
						size,
					},
				)
				.map_err(StageError::Vulkan)?;

				for node in scene.nodes.iter() {
					let model = self.load_model(device, ctx, delete, system, node.model)?;

					for mesh in model.meshes.iter() {
						let instance = instances.len() as u32;
						instances.push(Instance {
							transform: node.transform.cols.map(|x| x.xyz()),
							mesh: mesh.buffer.inner.id().unwrap(),
							material: mesh.material,
							_pad: [0; 2],
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
					tlas,
				};

				scene
					.instances
					.push(ctx, delete, cast_slice(&instances))
					.map_err(StageError::Vulkan)?;
				scene
					.meshlet_pointers
					.push(ctx, delete, cast_slice(&meshlet_pointers))
					.map_err(StageError::Vulkan)?;

				Ok::<_, StageError<_>>((Arc::new(scene), build_size))
			})
			.map_err(|x| match x {
				StageError::User(x) => match x {
					StageError::User(x) => StageError::User(x),
					StageError::Vulkan(x) => StageError::Vulkan(x),
				},
				StageError::Vulkan(x) => StageError::Vulkan(x),
			})?;

		let (_, ticket) = core
			.stage_after_ticket(device, ticket, |ctx, delete| unsafe {
				let buf = ctx.execute_with(QueueType::Graphics).map_err(StageError::Vulkan)?;
				let scratch = GpuBuffer::create(
					device,
					BufferDesc {
						size: build_size,
						usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
					},
				)
				.map_err(StageError::Vulkan)?;
				delete.delete(scratch);
				Ok::<_, StageError<_>>(())
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

	pub unsafe fn destroy(self, device: &CoreDevice) {
		self.material_buf.destroy(device);

		let err = "Cannot destroy `AssetRuntime` with asset references still alive";

		for (_, scene) in self.scenes {
			let scene = Arc::try_unwrap(scene).ok().expect(err);
			scene.instances.destroy(device);
			scene.meshlet_pointers.destroy(device);
			scene.tlas.destroy(device);
			for model in scene.models {
				if let Ok(model) = Arc::try_unwrap(model) {
					model.blas.destroy(device);
				}
			}
		}

		for (_, mesh) in self.meshes {
			let mesh = Arc::try_unwrap(mesh).ok().expect(err);
			mesh.buffer.destroy(device);
		}

		for (_, data) in self.images {
			let data = Arc::try_unwrap(data).ok().expect(err);
			data.view.destroy(device);
			data.image.destroy(device);
		}
	}

	fn load_model<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, queue: &mut DeletionQueue, system: &AssetSystem<S>,
		model: Uuid,
	) -> Result<Arc<Model>, StageError<AssetError<S>>> {
		if let Some(m) = self.models.get(&model) {
			Ok(m.clone())
		} else {
			let m = match system.load(model)? {
				Asset::Model(m) => m,
				_ => unreachable!("Model asset is not a model"),
			};
			let mut vertex_count = 0;
			let mut prim_count = 0;
			let meshes = m
				.meshes
				.into_iter()
				.map(|mesh| {
					let mesh = self.load_mesh(device, ctx, queue, system, mesh)?;
					vertex_count += mesh.vertex_count;
					prim_count += mesh.tri_count;
					Ok::<_, StageError<AssetError<S>>>(mesh)
				})
				.collect::<Result<Vec<_>, _>>()?;

			let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;
			let (size, build_size) = unsafe {
				let s = device.as_ext().get_acceleration_structure_build_sizes(
					vk::AccelerationStructureBuildTypeKHR::DEVICE,
					&vk::AccelerationStructureBuildGeometryInfoKHR::builder()
						.ty(ty)
						.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
						.geometries(&[vk::AccelerationStructureGeometryKHR::builder()
							.flags(vk::GeometryFlagsKHR::empty())
							.geometry_type(vk::GeometryTypeKHR::TRIANGLES)
							.geometry(vk::AccelerationStructureGeometryDataKHR {
								triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
									.vertex_format(vk::Format::R32G32B32_SFLOAT)
									.vertex_stride(std::mem::size_of::<Vec3<f32>>() as _)
									.max_vertex(vertex_count)
									.index_type(vk::IndexType::UINT32)
									.build(),
							})
							.build()]),
					&[prim_count],
				);
				(s.acceleration_structure_size, s.build_scratch_size)
			};
			let blas = AS::create(
				device,
				ASDesc {
					ty,
					flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
					size,
				},
			)
			.map_err(StageError::Vulkan)?;

			let m = Arc::new(Model {
				meshes,
				aabb: m.aabb,
				blas,
			});

			self.models.insert(model, m.clone());
			Ok(m)
		}
	}

	fn load_mesh<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, queue: &mut DeletionQueue, system: &AssetSystem<S>,
		mesh: Uuid,
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

			let mut vertex_count = 0;
			let mut tri_count = 0;
			for m in m.meshlets.iter() {
				vertex_count += m.vert_count as u32;
				tri_count += m.tri_count as u32;
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

			let material = self.load_material(device, ctx, queue, system, m.material)?;

			let m = Arc::new(Mesh {
				buffer,
				meshlet_count: m.meshlets.len() as u32,
				material,
				vertex_count,
				tri_count,
			});
			self.meshes.insert(mesh, m.clone());
			Ok(m)
		}
	}
}
