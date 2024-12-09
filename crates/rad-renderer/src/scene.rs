use std::collections::BTreeMap;

use ash::vk;
use bytemuck::NoUninit;
use rad_core::Engine;
use rad_graph::{
	device::{Device, ShaderInfo},
	graph::{BufferDesc, BufferLoc, BufferUsage, BufferUsageType, ExternalBuffer, Frame, Res},
	resource::{self, Buffer, BufferHandle, GpuPtr, Resource},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};
use rad_world::{system::WorldId, transform::Transform, Entity};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use vek::{Aabb, Quaternion, Vec3};

use crate::{assets::mesh::GpuAabb, components::mesh::MeshComponent};

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
}

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
struct GpuNewInstance {
	pub transform: GpuTransform,
	pub aabb: GpuAabb,
	pub mesh: GpuPtr<u8>,
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
		&'pass self, frame: &mut Frame<'pass, '_>, scene: &'pass mut Scene, id: WorldId, frame_index: u64,
	) -> SceneReader {
		let Scene {
			instances,
			len,
			cap,
			updates,
			depth_refs,
			..
		} = scene;

		let res = if *len > *cap {
			while *len > *cap {
				*cap *= 2;
			}

			let new = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "scene instances",
					size: *cap as u64 * std::mem::size_of::<GpuInstance>() as u64,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER
						| vk::BufferUsageFlags::TRANSFER_SRC
						| vk::BufferUsageFlags::TRANSFER_DST,
					readback: false,
				},
			)
			.unwrap();
			let old = std::mem::replace(instances, new);

			let mut pass = frame.pass("copy scene");
			let src = pass.resource(
				ExternalBuffer { handle: old.handle() },
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
			pass.build(move |mut pass| unsafe {
				let src = pass.get(src);
				let dst = pass.get(dst);
				pass.device.device().cmd_copy_buffer(
					pass.buf,
					src.buffer,
					dst.buffer,
					&[vk::BufferCopy::default()
						.src_offset(0)
						.dst_offset(0)
						.size(*cap as u64 * std::mem::size_of::<GpuInstance>() as u64)],
				);
			});
			frame.delete(old);

			Some(dst)
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
		let instances = if let Some(instances) = res {
			pass.reference(instances, usages);
			instances
		} else {
			pass.resource(
				ExternalBuffer {
					handle: instances.handle(),
				},
				usages,
			)
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

		SceneReader {
			instances,
			instance_count: *len,
			max_depth: depth_refs.first_key_value().map(|(InvertOrd(d), _)| *d).unwrap_or(0),
			frame: frame_index,
			id,
		}
	}
}

#[derive(Copy, Clone)]
pub struct SceneReader {
	pub instances: Res<BufferHandle>,
	pub instance_count: u32,
	pub max_depth: u32,
	pub frame: u64,
	pub id: WorldId,
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
					usage: vk::BufferUsageFlags::STORAGE_BUFFER
						| vk::BufferUsageFlags::TRANSFER_SRC
						| vk::BufferUsageFlags::TRANSFER_DST,
					readback: false,
				},
			)?,
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
			},
		});
	}
}
