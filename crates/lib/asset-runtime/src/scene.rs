use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use radiance_asset::{scene, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{device::descriptor::BufferId, resource::BufferDesc};
use radiance_util::{buffer::AllocBuffer, deletion::IntoResource, staging::StageError};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Mat4, Vec3, Vec4};

use crate::{
	mesh::Mesh,
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LErr,
	LResult,
	Loader,
};

pub struct Node {
	pub name: String,
	pub transform: Mat4<f32>,
	pub mesh: RRef<Mesh>,
	pub instance: u32,
}

pub struct Scene {
	instance_buffer: AllocBuffer,
	meshlet_pointer_buffer: AllocBuffer,
	meshlet_pointer_count: u32,
	pub cameras: Vec<scene::Camera>,
	pub nodes: Vec<Node>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuMeshletPointer {
	pub instance: u32,
	pub meshlet: u32,
}

const_assert_eq!(std::mem::size_of::<GpuMeshletPointer>(), 8);
const_assert_eq!(std::mem::align_of::<GpuMeshletPointer>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuInstance {
	pub transform: Vec4<Vec3<f32>>,
	/// Mesh buffer containing meshlets + meshlet data.
	pub mesh: BufferId,
	pub submesh_count: u32,
}

const_assert_eq!(std::mem::size_of::<GpuInstance>(), 56);
const_assert_eq!(std::mem::align_of::<GpuInstance>(), 4);

impl RuntimeAsset for Scene {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(self.instance_buffer.into_resource().into()).unwrap();
		queue.send(self.meshlet_pointer_buffer.into_resource().into()).unwrap();
	}
}

impl Scene {
	pub fn instances(&self) -> BufferId { self.instance_buffer.id().unwrap() }

	pub fn meshlet_pointers(&self) -> BufferId { self.meshlet_pointer_buffer.id().unwrap() }

	pub fn meshlet_pointer_count(&self) -> u32 { self.meshlet_pointer_count }
}

impl AssetRuntime {
	pub fn load_scene_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, '_, '_, S>, scene: Uuid,
	) -> LResult<Scene, S> {
		let Asset::Scene(s) = loader.sys.load(scene)? else {
			unreachable!("Scene asset is not a scene");
		};

		let size = (std::mem::size_of::<GpuInstance>() * s.nodes.len()) as u64;
		let mut instance_buffer = AllocBuffer::new(
			loader.device,
			BufferDesc {
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			},
		)
		.map_err(StageError::Vulkan)?;
		instance_buffer
			.alloc_size(loader.ctx, loader.queue, size)
			.map_err(StageError::Vulkan)?;
		let mut writer = SliceWriter::new(unsafe { instance_buffer.data().as_mut() });

		let nodes: Vec<_> = s
			.nodes
			.into_iter()
			.enumerate()
			.map(|(i, n)| {
				let mesh = self.load_mesh(loader, n.model)?;
				writer
					.write(GpuInstance {
						transform: n.transform.cols.map(|x| x.xyz()),
						mesh: mesh.buffer.id().unwrap(),
						submesh_count: mesh.submeshes.len() as u32,
					})
					.unwrap();
				Ok(Node {
					name: n.name,
					transform: n.transform,
					mesh,
					instance: i as u32,
				})
			})
			.collect::<Result<_, LErr<S>>>()?;

		let meshlet_pointer_count: u32 = nodes.iter().map(|n| n.mesh.meshlet_count).sum();
		let size = std::mem::size_of::<GpuMeshletPointer>() as u64 * meshlet_pointer_count as u64;
		let mut meshlet_pointer_buffer = AllocBuffer::new(
			loader.device,
			BufferDesc {
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			},
		)
		.map_err(StageError::Vulkan)?;
		meshlet_pointer_buffer
			.alloc_size(loader.ctx, loader.queue, size)
			.map_err(StageError::Vulkan)?;
		let mut writer = SliceWriter::new(unsafe { meshlet_pointer_buffer.data().as_mut() });

		for (instance, node) in nodes.iter().enumerate() {
			for meshlet in 0..node.mesh.meshlet_count {
				writer
					.write(GpuMeshletPointer {
						instance: instance as u32,
						meshlet,
					})
					.unwrap();
			}
		}

		Ok(RRef::new(
			Scene {
				nodes,
				instance_buffer,
				meshlet_pointer_buffer,
				meshlet_pointer_count,
				cameras: s.cameras,
			},
			loader.deleter.clone(),
		))
	}
}

