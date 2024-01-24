use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use radiance_asset::{mesh::Vertex, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	device::descriptor::BufferId,
	resource::{BufferDesc, GpuBuffer, Resource},
};
use radiance_util::{deletion::IntoResource, staging::StageError};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Vec3, Vec4};

use crate::{
	material::Material,
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LErr,
	LResult,
	Loader,
};

pub type GpuVertex = Vertex;

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
pub struct GpuMeshlet {
	pub aabb_min: Vec3<f32>,
	pub aabb_extent: Vec3<f32>,
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	pub submesh: u16,
}

const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 36);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuSubMesh {
	pub material: u32,
}

const_assert_eq!(std::mem::size_of::<GpuSubMesh>(), 4);
const_assert_eq!(std::mem::align_of::<GpuSubMesh>(), 4);

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

pub struct Mesh {
	pub buffer: GpuBuffer,
	pub submeshes: Vec<RRef<Material>>,
	pub meshlet_count: u32,
}

impl RuntimeAsset for Mesh {
	fn into_resources(self, queue: Sender<DelRes>) { queue.send(self.buffer.into_resource().into()).unwrap(); }
}

impl AssetRuntime {
	pub(crate) fn load_mesh_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, '_, '_, S>, mesh: Uuid,
	) -> LResult<Mesh, S> {
		let Asset::Mesh(m) = loader.sys.load(mesh)? else {
			unreachable!("Mesh asset is not a mesh");
		};

		let submesh_byte_offset = 0;
		let submesh_byte_len = (m.submeshes.len() * std::mem::size_of::<GpuSubMesh>()) as u64;
		let meshlet_byte_offset = submesh_byte_offset + submesh_byte_len;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() / 3 * std::mem::size_of::<u32>()) as u64;
		let size = index_byte_offset + index_byte_len;

		let meshlet_count = m.meshlets.len() as u32;

		let buffer = GpuBuffer::create(
			loader.device,
			BufferDesc {
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			},
		)
		.map_err(StageError::Vulkan)?;

		let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });
		let submeshes = m
			.submeshes
			.iter()
			.map(|x| {
				let mat = self.load_material(loader, x.material)?;
				writer.write(GpuSubMesh { material: mat.index }).unwrap();
				Ok(mat)
			})
			.collect::<Result<_, LErr<S>>>()?;

		let mut srs = m.submeshes.into_iter().map(|x| x.meshlets);
		let mut curr = srs.next().unwrap();
		let mut curr_i = 0;
		for (i, m) in m.meshlets.into_iter().enumerate() {
			let submesh = if curr.contains(&(i as u32)) {
				curr_i
			} else {
				curr = srs.next().unwrap();
				curr_i += 1;
				curr_i
			};
			writer
				.write(GpuMeshlet {
					aabb_min: m.aabb.min,
					aabb_extent: m.aabb.max - m.aabb.min,
					vertex_byte_offset: vertex_byte_offset as u32
						+ (m.vertex_offset * std::mem::size_of::<GpuVertex>() as u32),
					index_byte_offset: index_byte_offset as u32
						+ (m.index_offset / 3 * std::mem::size_of::<u32>() as u32),
					vertex_count: m.vert_count,
					triangle_count: m.tri_count,
					submesh,
				})
				.unwrap();
		}

		writer.write_slice(&m.vertices).unwrap();

		for tri in m.indices.chunks(3) {
			writer.write_slice(tri).unwrap();
			writer.write(0u8).unwrap();
		}

		Ok(RRef::new(
			Mesh {
				buffer,
				submeshes,
				meshlet_count,
			},
			loader.deleter.clone(),
		))
	}
}

