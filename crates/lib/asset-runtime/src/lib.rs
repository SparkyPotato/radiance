#![feature(allocator_api)]

//! Bridge between raw assets and cached assets on the GPU or CPU.
//!
//! TODO: This entire crate/file is absolutely horrible and needs a rework.

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
	resource::{BufferDesc, GpuBuffer, Image, ImageDesc, ImageView, ImageViewDesc, ImageViewUsage, Resource},
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
	pub material: u32,
	pub _pad: [u32; 2],
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

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct Material {
	pub base_color_factor: Vec4<f32>,
	pub base_color: Option<ImageId>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	pub metallic_roughness: Option<ImageId>,
	pub normal: Option<ImageId>,
	pub occlusion: Option<ImageId>,
	pub emissive_factor: Vec3<f32>,
	pub emissive: Option<ImageId>,
}

const_assert_eq!(std::mem::size_of::<Material>(), 56);
const_assert_eq!(std::mem::align_of::<Material>(), 4);

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

impl Scene {
	pub fn meshlet_count(&self) -> u32 {
		(self.meshlet_pointers.len() / std::mem::size_of::<MeshletPointer>() as u64) as u32
	}
}

pub struct Mesh {
	pub buffer: GpuBuffer,
	pub meshlet_count: u32,
	pub material: u32,
}

pub struct ImageData {
	pub image: Image,
	pub view: ImageView,
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

		let (scene, ticket) = core
			.stage(device, |ctx, delete| {
				let mut instances = Vec::with_capacity(scene.nodes.len());
				let mut meshlet_pointers = Vec::with_capacity(scene.nodes.len() * 8);
				let mut models = Vec::with_capacity(scene.nodes.len());

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
		self.material_buf.destroy(device);

		let err = "Cannot destroy `AssetRuntime` with asset references still alive";

		for (_, scene) in self.scenes {
			let scene = Arc::try_unwrap(scene).ok().expect(err);
			scene.instances.destroy(device);
			scene.meshlet_pointers.destroy(device);
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
			let meshes = m
				.meshes
				.into_iter()
				.map(|mesh| self.load_mesh(device, ctx, queue, system, mesh))
				.collect::<Result<Vec<_>, _>>()?;

			let m = Arc::new(Model { meshes, aabb: m.aabb });

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

			let material = self.load_material(device, ctx, queue, system, m.material)?;

			let m = Arc::new(Mesh {
				buffer,
				meshlet_count: m.meshlets.len() as u32,
				material,
			});
			self.meshes.insert(mesh, m.clone());
			Ok(m)
		}
	}

	fn load_material<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, queue: &mut DeletionQueue, system: &AssetSystem<S>,
		material: Uuid,
	) -> Result<u32, StageError<AssetError<S>>> {
		if let Some(&id) = self.materials.get(&material) {
			Ok(id)
		} else {
			let Asset::Material(m) = system.load(material)? else {
				unreachable!("Material asset is not a material");
			};

			let mut load = |image, srgb| self.load_image(device, ctx, system, image, srgb);
			let base_color = m
				.base_color
				.map(|x| load(x, true))
				.transpose()?
				.map(|x| x.view.id.unwrap());
			let metallic_roughness = m
				.metallic_roughness
				.map(|x| load(x, false))
				.transpose()?
				.map(|x| x.view.id.unwrap());
			let normal = m
				.normal
				.map(|x| load(x, false))
				.transpose()?
				.map(|x| x.view.id.unwrap());
			let occlusion = m
				.occlusion
				.map(|x| load(x, false))
				.transpose()?
				.map(|x| x.view.id.unwrap());
			let emissive = m
				.emissive
				.map(|x| load(x, false))
				.transpose()?
				.map(|x| x.view.id.unwrap());
			let mat = Material {
				base_color_factor: m.base_color_factor,
				base_color,
				metallic_factor: m.metallic_factor,
				roughness_factor: m.roughness_factor,
				metallic_roughness,
				normal,
				occlusion,
				emissive_factor: m.emissive_factor,
				emissive,
			};
			let offset = self
				.material_buf
				.push(ctx, queue, bytes_of(&mat))
				.map_err(StageError::Vulkan)?;
			let index = (offset / std::mem::size_of::<Material>() as u64) as u32;
			self.materials.insert(material, index);
			Ok(index)
		}
	}

	fn load_image<S: AssetSource>(
		&mut self, device: &CoreDevice, ctx: &mut StagingCtx, system: &AssetSystem<S>, image: Uuid, srgb: bool,
	) -> Result<Arc<ImageData>, StageError<AssetError<S>>> {
		if let Some(x) = self.images.get(&image) {
			return Ok(x.clone());
		} else {
			let Asset::Image(i) = system.load(image)? else {
				unreachable!("image asset is not image");
			};

			let format = match i.format {
				Format::R8 => {
					if srgb {
						vk::Format::R8_SRGB
					} else {
						vk::Format::R8_UNORM
					}
				},
				Format::R8G8 => {
					if srgb {
						vk::Format::R8G8_SRGB
					} else {
						vk::Format::R8G8_UNORM
					}
				},
				Format::R8G8B8A8 => {
					if srgb {
						vk::Format::R8G8B8A8_SRGB
					} else {
						vk::Format::R8G8B8A8_UNORM
					}
				},
				Format::R16 => vk::Format::R16_UNORM,
				Format::R16G16 => vk::Format::R16G16_UNORM,
				Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
				Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
				Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
				Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
			};
			let size = vk::Extent3D::builder().width(i.width).height(i.height).depth(1).build();
			let img = Image::create(
				device,
				ImageDesc {
					flags: vk::ImageCreateFlags::empty(),
					format,
					size,
					levels: 1,
					layers: 1,
					samples: vk::SampleCountFlags::TYPE_1,
					usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
				},
			)
			.map_err(StageError::Vulkan)?;
			let view = ImageView::create(
				device,
				ImageViewDesc {
					image: img.handle(),
					view_type: vk::ImageViewType::TYPE_2D,
					format,
					usage: ImageViewUsage::Sampled,
					aspect: vk::ImageAspectFlags::COLOR,
					size,
				},
			)
			.map_err(StageError::Vulkan)?;
			ctx.stage_image(
				&i.data,
				img.handle(),
				ImageStage {
					buffer_row_length: 0,
					buffer_image_height: 0,
					image_subresource: vk::ImageSubresourceLayers::builder()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.mip_level(0)
						.base_array_layer(0)
						.layer_count(1)
						.build(),
					image_offset: vk::Offset3D::default(),
					image_extent: size,
				},
				true,
				QueueType::Graphics,
				&[],
				&[ImageUsage::ShaderReadSampledImage(Shader::Any)],
			)
			.map_err(StageError::Vulkan)?;

			let data = Arc::new(ImageData { image: img, view });
			self.images.insert(image, data.clone());
			Ok(data)
		}
	}
}

