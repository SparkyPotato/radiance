use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use radiance_asset::{scene, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	device::{
		descriptor::{ASId, BufferId},
		QueueType,
	},
	resource::{ASDesc, BufferDesc, GpuBuffer, Resource, AS},
	sync::{get_global_barrier, GlobalBarrier, UsageType},
};
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
	acceleration_structure: AS,
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
	pub meshlet_count: u32,
	pub submesh_count: u32,
}

const_assert_eq!(std::mem::size_of::<GpuInstance>(), 60);
const_assert_eq!(std::mem::align_of::<GpuInstance>(), 4);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkAccelerationStructureInstanceKHR {
	pub transform: vk::TransformMatrixKHR,
	pub instance_custom_index_and_mask: vk::Packed24_8,
	pub instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8,
	pub acceleration_structure_reference: vk::AccelerationStructureReferenceKHR,
}

unsafe impl NoUninit for VkAccelerationStructureInstanceKHR {}

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

	pub fn acceleration_structure(&self) -> ASId { self.acceleration_structure.id.unwrap() }
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

		let temp_build_buffer = GpuBuffer::create(
			loader.device,
			BufferDesc {
				size: (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * s.nodes.len()) as u64,
				usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
			},
		)
		.map_err(StageError::Vulkan)?;
		let mut awriter = SliceWriter::new(unsafe { temp_build_buffer.data().as_mut() });

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
						meshlet_count: mesh.meshlet_count,
						submesh_count: mesh.submeshes.len() as u32,
					})
					.unwrap();
				awriter
					.write(VkAccelerationStructureInstanceKHR {
						transform: vk::TransformMatrixKHR {
							matrix: unsafe {
								std::mem::transmute(
									n.transform.transposed().cols.xyz().map(|x| x.into_array()).into_array(),
								)
							},
						},
						instance_custom_index_and_mask: vk::Packed24_8::new(i as _, 0xff),
						instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
						acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
							device_handle: unsafe {
								loader.device.as_ext().get_acceleration_structure_device_address(
									&vk::AccelerationStructureDeviceAddressInfoKHR::builder()
										.acceleration_structure(mesh.acceleration_structure.handle()),
								)
							},
						},
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

		let acceleration_structure = unsafe {
			let ext = loader.device.as_ext();

			let geo = [vk::AccelerationStructureGeometryKHR::builder()
				.geometry_type(vk::GeometryTypeKHR::INSTANCES)
				.geometry(vk::AccelerationStructureGeometryDataKHR {
					instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
						.array_of_pointers(false)
						.data(vk::DeviceOrHostAddressConstKHR {
							device_address: temp_build_buffer.addr(),
						})
						.build(),
				})
				.flags(vk::GeometryFlagsKHR::OPAQUE)
				.build()];
			let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
				.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
				.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
				.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
				.geometries(&geo);

			let count = nodes.len() as u32;
			let size = ext.get_acceleration_structure_build_sizes(
				vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&info,
				&[count],
			);

			let as_ = AS::create(
				loader.device,
				ASDesc {
					flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
					ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
					size: size.acceleration_structure_size,
				},
			)
			.map_err(StageError::Vulkan)?;

			let scratch = GpuBuffer::create(
				loader.device,
				BufferDesc {
					size: size.build_scratch_size,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				},
			)
			.map_err(StageError::Vulkan)?;

			info.dst_acceleration_structure = as_.handle();
			info.scratch_data = vk::DeviceOrHostAddressKHR {
				device_address: scratch.addr(),
			};

			let buf = loader
				.ctx
				.execute_before(QueueType::Compute)
				.map_err(StageError::Vulkan)?;

			loader.device.device().cmd_pipeline_barrier2(
				buf,
				&vk::DependencyInfo::builder().memory_barriers(&[get_global_barrier(&GlobalBarrier {
					previous_usages: &[UsageType::AccelerationStructureBuildWrite],
					next_usages: &[UsageType::AccelerationStructureBuildRead],
				})]),
			);
			ext.cmd_build_acceleration_structures(
				buf,
				&[info.build()],
				&[&[vk::AccelerationStructureBuildRangeInfoKHR::builder()
					.primitive_count(count)
					.build()]],
			);

			loader.queue.delete(temp_build_buffer);
			loader.queue.delete(scratch);

			as_
		};

		Ok(RRef::new(
			Scene {
				nodes,
				instance_buffer,
				meshlet_pointer_buffer,
				acceleration_structure,
				meshlet_pointer_count,
				cameras: s.cameras,
			},
			loader.deleter.clone(),
		))
	}
}

