use std::collections::BTreeMap;

use ash::vk;
use bytemuck::NoUninit;
use rad_core::Engine;
use rad_graph::{
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, ExternalBuffer, Frame, Res},
	resource::{self, ASDesc, Buffer, BufferHandle, GpuPtr, Resource, AS},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};
use rad_world::{transform::Transform, Entity};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use vek::{Aabb, Quaternion, Vec3};

use crate::{
	assets::mesh::{GpuAabb, GpuVertex},
	components::mesh::MeshComponent,
};

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
pub struct GpuTransform {
	pub position: Vec3<f32>,
	pub rotation: Quaternion<f32>,
	pub scale: Vec3<f32>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuInstance {
	pub transform: GpuTransform,
	pub prev_transform: GpuTransform,
	pub aabb: GpuAabb,
	pub update_frame: u64,
	pub mesh: GpuPtr<u8>,
	pub raw_mesh: GpuPtr<GpuVertex>,
}

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
struct GpuNewInstance {
	transform: GpuTransform,
	aabb: GpuAabb,
	mesh: GpuPtr<u8>,
	raw_mesh: GpuPtr<GpuVertex>,
	as_: u64,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(u32)]
enum GpuUpdateType {
	Add,
	Move,
	ChangeMesh,
	ChangeTransform,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct GpuSceneUpdate {
	pub instance: u32,
	pub ty: GpuUpdateType,
	pub data: GpuNewInstance,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	as_instances: GpuPtr<u64>,
	updates: GpuPtr<GpuSceneUpdate>,
	frame: u64,
	count: u32,
	_pad: u32,
}

pub struct SceneUpdater {
	pass: ComputePass<PushConstants>,
}

impl SceneUpdater {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			pass: ComputePass::new(
				device,
				ShaderInfo {
					shader: "asset.scene.update",
					spec: &[],
				},
			)?,
		})
	}

	pub fn update<'pass>(
		&'pass self, frame: &mut Frame<'pass, '_>, scene: &'pass mut Scene, frame_index: u64,
	) -> SceneReader {
		let Scene {
			instances,
			as_instances,
			as_,
			len,
			cap,
			updates,
			depth_refs,
			..
		} = scene;
		let asi = as_instances.handle();

		let res = if *len > *cap {
			while *len > *cap {
				*cap *= 2;
			}

			let new = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "scene instances",
					size: *cap as u64 * std::mem::size_of::<GpuInstance>() as u64,
					readback: false,
				},
			)
			.unwrap();
			let new_as = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "AS scene instances",
					size: *cap as u64 * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
					readback: false,
				},
			)
			.unwrap();
			let old = std::mem::replace(instances, new);
			let old_as = std::mem::replace(as_instances, new_as);

			let mut pass = frame.pass("copy scene");
			let src = pass.resource(
				ExternalBuffer { handle: old.handle() },
				BufferUsage {
					usages: &[BufferUsageType::TransferRead],
				},
			);
			let src_as = pass.resource(
				ExternalBuffer {
					handle: old_as.handle(),
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferRead],
				},
			);
			let dst = pass.resource(
				ExternalBuffer {
					handle: instances.handle(),
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			);
			let dst_as = pass.resource(
				ExternalBuffer {
					handle: as_instances.handle(),
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			);
			pass.build(move |mut pass| unsafe {
				let src = pass.get(src);
				let src_as = pass.get(src_as);
				let dst = pass.get(dst);
				let dst_as = pass.get(dst_as);
				pass.device.device().cmd_copy_buffer(
					pass.buf,
					src.buffer,
					dst.buffer,
					&[vk::BufferCopy::default()
						.src_offset(0)
						.dst_offset(0)
						.size(*cap as u64 * std::mem::size_of::<GpuInstance>() as u64)],
				);
				pass.device.device().cmd_copy_buffer(
					pass.buf,
					src_as.buffer,
					dst_as.buffer,
					&[vk::BufferCopy::default()
						.src_offset(0)
						.dst_offset(0)
						.size(*cap as u64 * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64)],
				);
			});
			frame.delete(old);
			frame.delete(old_as);

			Some((dst, dst_as))
		} else {
			None
		};

		let mut pass = frame.pass("update scene");

		let count = updates.len();
		let usages = BufferUsage {
			usages: if count > 0 {
				&[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				]
			} else {
				&[]
			},
		};
		let (instances, as_instances) = if let Some((instances, as_instances)) = res {
			pass.reference(instances, usages);
			pass.reference(as_instances, usages);
			(instances, as_instances)
		} else {
			let i = pass.resource(
				ExternalBuffer {
					handle: instances.handle(),
				},
				usages,
			);
			let a = pass.resource(
				ExternalBuffer {
					handle: as_instances.handle(),
				},
				usages,
			);
			(i, a)
		};
		let update_buffer = (count > 0).then(|| {
			pass.resource(
				BufferDesc {
					size: (count * std::mem::size_of::<GpuSceneUpdate>()) as _,
					loc: BufferLoc::Upload,
					persist: None,
				},
				usages,
			)
		});

		let count = count as _;
		pass.build(move |mut pass| unsafe {
			if let Some(update_buffer) = update_buffer {
				let update_buf = pass.get(update_buffer);
				let ptr = Huh(update_buf.data.as_ptr() as _);
				updates.par_drain(..).enumerate().for_each(|(i, u)| {
					let ptr = &ptr;
					ptr.0.add(i).write(u);
				});

				self.pass.dispatch(
					&PushConstants {
						instances: pass.get(instances).ptr(),
						as_instances: pass.get(as_instances).ptr(),
						updates: update_buf.ptr(),
						frame: frame_index,
						count,
						_pad: 0,
					},
					&pass,
					(count + 63) / 64,
					1,
					1,
				);

				struct Huh(*mut GpuSceneUpdate);
				unsafe impl Send for Huh {}
				unsafe impl Sync for Huh {}
			}
		});

		let count = *len;
		let geo = [vk::AccelerationStructureGeometryKHR::default()
			.geometry_type(vk::GeometryTypeKHR::INSTANCES)
			.geometry(vk::AccelerationStructureGeometryDataKHR {
				instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
					.array_of_pointers(false)
					.data(vk::DeviceOrHostAddressConstKHR {
						device_address: asi.ptr::<u8>().addr(),
					}),
			})];
		let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
			.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
			.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
			.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
			.geometries(&geo);
		let mut sinfo = vk::AccelerationStructureBuildSizesInfoKHR::default();
		unsafe {
			frame.device().as_ext().get_acceleration_structure_build_sizes(
				vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&info,
				&[count],
				&mut sinfo,
			);
		}

		let mut curr_size = as_.size();
		if sinfo.acceleration_structure_size > curr_size {
			if curr_size == 0 {
				curr_size = 1024;
			}
			while sinfo.acceleration_structure_size > curr_size {
				curr_size *= 2;
			}
			let old = std::mem::replace(
				as_,
				AS::create(
					frame.device(),
					ASDesc {
						name: "tlas",
						flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
						ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
						size: curr_size,
					},
				)
				.unwrap(),
			);
			frame.delete(old);
		}

		let mut pass = frame.pass("build tlas");
		pass.reference(
			as_instances,
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildRead],
			},
		);
		let scratch = pass.resource(
			BufferDesc {
				size: sinfo.build_scratch_size,
				loc: BufferLoc::GpuOnly,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildScratch],
			},
		);
		pass.resource(
			ExternalBuffer {
				handle: as_.buf_handle(),
			},
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildWrite],
			},
		);
		pass.build(move |mut pass| unsafe {
			let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
				.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
				.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
				.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
				.geometries(&geo);
			info.dst_acceleration_structure = as_.handle();
			info.scratch_data.device_address = pass.get(scratch).ptr::<u8>().addr();
			pass.device.as_ext().cmd_build_acceleration_structures(
				pass.buf,
				&[info],
				&[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
					.primitive_count(count)
					.primitive_offset(0)]],
			);
		});

		SceneReader {
			instances,
			instance_count: *len,
			max_depth: depth_refs.first_key_value().map(|(InvertOrd(d), _)| *d).unwrap_or(0),
			frame: frame_index,
		}
	}
}

#[derive(Copy, Clone)]
pub struct SceneReader {
	pub instances: Res<BufferHandle>,
	pub instance_count: u32,
	pub max_depth: u32,
	pub frame: u64,
}

struct InvertOrd<T>(T);
impl<T: PartialOrd> PartialOrd for InvertOrd<T> {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { other.0.partial_cmp(&self.0) }
}
impl<T: Ord> Ord for InvertOrd<T> {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering { other.0.cmp(&self.0) }
}
impl<T: PartialEq> PartialEq for InvertOrd<T> {
	fn eq(&self, other: &Self) -> bool { other.0.eq(&self.0) }
}
impl<T: Eq> Eq for InvertOrd<T> {}

pub struct Scene {
	instances: Buffer,
	as_instances: Buffer,
	as_: AS,
	len: u32,
	cap: u32,
	entity_map: FxHashMap<Entity, (u32, u32)>,
	updates: Vec<GpuSceneUpdate>,
	depth_refs: BTreeMap<InvertOrd<u32>, u32>,
}

fn map_aabb(aabb: Aabb<f32>) -> GpuAabb {
	GpuAabb {
		center: aabb.center(),
		half_extent: aabb.half_size().into(),
	}
}

fn map_transform(transform: &Transform) -> GpuTransform {
	GpuTransform {
		position: transform.position,
		rotation: transform.rotation,
		scale: transform.scale,
	}
}

impl Scene {
	pub fn new() -> Result<Self> {
		let device: &Device = Engine::get().global();

		Ok(Self {
			instances: Buffer::create(
				device,
				resource::BufferDesc {
					name: "scene instances",
					size: std::mem::size_of::<GpuInstance>() as u64 * 1024,
					readback: false,
				},
			)?,
			as_instances: Buffer::create(
				device,
				resource::BufferDesc {
					name: "AS scene instances",
					size: std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64 * 1024,
					readback: false,
				},
			)?,
			as_: AS::default(),
			len: 0,
			cap: 1024,
			entity_map: FxHashMap::default(),
			updates: Vec::new(),
			depth_refs: BTreeMap::new(),
		})
	}

	pub fn add(&mut self, entity: Entity, transform: &Transform, mesh: &MeshComponent) {
		self.updates.push(GpuSceneUpdate {
			instance: self.len,
			ty: GpuUpdateType::Add,
			data: GpuNewInstance {
				transform: map_transform(transform),
				aabb: map_aabb(mesh.inner.aabb()),
				mesh: mesh.inner.gpu_ptr(),
				raw_mesh: mesh.inner.raw_gpu_ptr(),
				as_: mesh.inner.as_addr(),
			},
		});
		self.len += 1;

		let depth = mesh.inner.bvh_depth();
		*self.depth_refs.entry(InvertOrd(depth)).or_insert(0) += 1;
		self.entity_map.insert(entity, (self.len, depth));
	}

	pub fn remove(&mut self, entity: Entity) {
		let (instance, depth) = self.entity_map.remove(&entity).expect("entity not in scene");
		let depth = self.depth_refs.get_mut(&InvertOrd(depth)).unwrap();
		*depth -= 1;
		if *depth == 0 {
			let d = *depth;
			self.depth_refs.remove(&InvertOrd(d));
		}

		self.updates.push(GpuSceneUpdate {
			instance,
			ty: GpuUpdateType::Move,
			data: GpuNewInstance {
				transform: GpuTransform {
					position: Vec3::new(f32::from_bits(self.len - 1), 0.0, 0.0),
					..Default::default()
				},
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				as_: 0,
			},
		});
		self.len -= 1;
	}

	pub fn change_mesh_and_transform(&mut self, entity: Entity, transform: &Transform, mesh: &MeshComponent) {
		let (instance, depth) = self.entity_map.get_mut(&entity).expect("entity not in scene");
		let old_depth = self.depth_refs.get_mut(&InvertOrd(*depth)).unwrap();
		*old_depth -= 1;
		if *old_depth == 0 {
			self.depth_refs.remove(&InvertOrd(*depth));
		}

		self.updates.push(GpuSceneUpdate {
			instance: *instance,
			ty: GpuUpdateType::Add,
			data: GpuNewInstance {
				transform: map_transform(transform),
				aabb: map_aabb(mesh.inner.aabb()),
				mesh: mesh.inner.gpu_ptr(),
				raw_mesh: mesh.inner.raw_gpu_ptr(),
				as_: mesh.inner.as_addr(),
			},
		});

		let new_depth = mesh.inner.bvh_depth();
		*self.depth_refs.entry(InvertOrd(new_depth)).or_insert(0) += 1;
		*depth = new_depth;
	}

	pub fn change_mesh(&mut self, entity: Entity, mesh: &MeshComponent) {
		let (instance, depth) = self.entity_map.get_mut(&entity).expect("entity not in scene");
		let old_depth = self.depth_refs.get_mut(&InvertOrd(*depth)).unwrap();
		*old_depth -= 1;
		if *old_depth == 0 {
			self.depth_refs.remove(&InvertOrd(*depth));
		}

		self.updates.push(GpuSceneUpdate {
			instance: *instance,
			ty: GpuUpdateType::ChangeMesh,
			data: GpuNewInstance {
				transform: GpuTransform::default(),
				aabb: map_aabb(mesh.inner.aabb()),
				mesh: mesh.inner.gpu_ptr(),
				raw_mesh: mesh.inner.raw_gpu_ptr(),
				as_: mesh.inner.as_addr(),
			},
		});

		let new_depth = mesh.inner.bvh_depth();
		*self.depth_refs.entry(InvertOrd(new_depth)).or_insert(0) += 1;
		*depth = new_depth;
	}

	pub fn change_transform(&mut self, entity: Entity, transform: &Transform) {
		let (instance, _) = self.entity_map.get(&entity).expect("entity not in scene");

		self.updates.push(GpuSceneUpdate {
			instance: *instance,
			ty: GpuUpdateType::ChangeTransform,
			data: GpuNewInstance {
				transform: map_transform(transform),
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				as_: 0,
			},
		});
	}
}
