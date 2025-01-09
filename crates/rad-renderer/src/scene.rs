use std::collections::BTreeMap;

use ash::vk;
use bytemuck::{checked::cast_slice_mut, NoUninit, Pod, Zeroable};
use rad_core::{
	asset::aref::{ARef, AWeak},
	Engine,
};
use rad_graph::{
	device::{descriptor::ImageId, Device, ShaderInfo},
	graph::{BufferDesc, BufferUsage, BufferUsageType, ExternalBuffer, Frame, Res},
	resource::{self, ASDesc, Buffer, BufferHandle, GpuPtr, Resource, AS},
	sync::Shader,
	util::compute::ComputePass,
	Result,
};
use rad_world::{transform::Transform, Entity};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use vek::{Aabb, Quaternion, Vec3, Vec4};

use crate::{
	assets::{
		image::Image,
		material::Material,
		mesh::{GpuAabb, GpuVertex, Mesh},
	},
	components::{
		light::{LightComponent, LightType},
		mesh::MeshComponent,
	},
};

#[derive(Copy, Clone, Default, PartialEq, NoUninit)]
#[repr(C)]
pub struct GpuTransform {
	pub position: Vec3<f32>,
	pub rotation: Quaternion<f32>,
	pub scale: Vec3<f32>,
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMaterial {
	pub base_color: Option<ImageId>,
	pub base_color_factor: Vec4<f32>,
	pub metallic_roughness: Option<ImageId>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	pub normal: Option<ImageId>,
	pub emissive: Option<ImageId>,
	pub emissive_factor: Vec3<f32>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(u32)]
pub enum GpuLightType {
	Point,
	Directional,
	Emissive,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuLight {
	pub ty: GpuLightType,
	pub radiance: Vec3<f32>,
	pub pos_or_dir: Vec3<f32>,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuInstance {
	transform: GpuTransform,
	prev_transform: GpuTransform,
	aabb: GpuAabb,
	update_frame: u64,
	mesh: GpuPtr<u8>,
	raw_mesh: GpuPtr<GpuVertex>,
	material: GpuPtr<GpuMaterial>,
	raw_vertex_count: u32,
	raw_tri_count: u32,
}

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
struct GpuUpdate {
	transform: GpuTransform, // for lights, position = radiance, scale = pos_or_dir
	aabb: GpuAabb,
	mesh: GpuPtr<u8>,
	raw_mesh: GpuPtr<GpuVertex>,
	material: GpuPtr<GpuMaterial>,
	as_: u64,
	raw_vertex_count: u32, // move => index, add light => ty
	raw_tri_count: u32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(u32)]
enum GpuUpdateType {
	Add,
	Move,
	ChangeMesh,
	ChangeTransform,
	AddLight,
	MoveLight,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct GpuSceneUpdate {
	pub instance: u32,
	pub ty: GpuUpdateType,
	pub data: GpuUpdate,
}

impl GpuSceneUpdate {
	fn add(instance: u32, transform: &Transform, m: &ARef<Mesh>, material: GpuPtr<GpuMaterial>) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::Add,
			data: GpuUpdate {
				transform: map_transform(transform),
				aabb: map_aabb(m.aabb()),
				mesh: m.gpu_ptr(),
				raw_mesh: m.raw_gpu_ptr(),
				as_: m.as_addr(),
				material,
				raw_vertex_count: m.raw_vertex_count(),
				raw_tri_count: m.raw_tri_count(),
			},
		}
	}

	fn remove(instance: u32, last: u32) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::Move,
			data: GpuUpdate {
				transform: GpuTransform::default(),
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				as_: 0,
				material: GpuPtr::null(),
				raw_vertex_count: last,
				raw_tri_count: 0,
			},
		}
	}

	fn change_mesh(instance: u32, m: &ARef<Mesh>, material: GpuPtr<GpuMaterial>) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::ChangeMesh,
			data: GpuUpdate {
				transform: GpuTransform::default(),
				aabb: map_aabb(m.aabb()),
				mesh: m.gpu_ptr(),
				raw_mesh: m.raw_gpu_ptr(),
				as_: m.as_addr(),
				material,
				raw_vertex_count: m.raw_vertex_count(),
				raw_tri_count: m.raw_tri_count(),
			},
		}
	}

	fn change_transform(instance: u32, transform: &Transform) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::ChangeTransform,
			data: GpuUpdate {
				transform: map_transform(transform),
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				as_: 0,
				material: GpuPtr::null(),
				raw_vertex_count: 0,
				raw_tri_count: 0,
			},
		}
	}

	fn add_light(instance: u32, transform: &Transform, light: &LightComponent) -> Self {
		let pos_or_dir = match light.ty {
			LightType::Point => transform.position,
			LightType::Directional => (transform.rotation * -Vec4::unit_z()).xyz(),
		};

		Self {
			instance,
			ty: GpuUpdateType::AddLight,
			data: GpuUpdate {
				transform: GpuTransform {
					position: light.radiance,
					rotation: Quaternion::zero(),
					scale: pos_or_dir,
				},
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				material: GpuPtr::null(),
				as_: 0,
				raw_vertex_count: map_light_ty(light.ty) as _,
				raw_tri_count: 0,
			},
		}
	}

	fn remove_light(instance: u32, last: u32) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::MoveLight,
			data: GpuUpdate {
				transform: GpuTransform::default(),
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				material: GpuPtr::null(),
				as_: 0,
				raw_vertex_count: last,
				raw_tri_count: 0,
			},
		}
	}

	fn add_emissive(instance: u32, mesh_instance: u32) -> Self {
		Self {
			instance,
			ty: GpuUpdateType::AddLight,
			data: GpuUpdate {
				transform: GpuTransform {
					position: Vec3::new(f32::from_bits(mesh_instance), 0.0, 0.0),
					..Default::default()
				},
				aabb: GpuAabb::default(),
				mesh: GpuPtr::null(),
				raw_mesh: GpuPtr::null(),
				material: GpuPtr::null(),
				as_: 0,
				raw_vertex_count: GpuLightType::Emissive as _,
				raw_tri_count: 0,
			},
		}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
struct PushConstants {
	instances: GpuPtr<GpuInstance>,
	as_instances: GpuPtr<u64>,
	lights: GpuPtr<GpuLight>,
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
			lights,
			as_instances,
			as_,
			len,
			cap,
			light_len,
			light_cap,
			updates,
			depth_refs,
			sun_dir,
			sun_radiance,
			..
		} = scene;
		let asi = as_instances.handle();

		let ins = if *len > *cap {
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

			let mut pass = frame.pass("copy meshes");
			let src = pass.resource(ExternalBuffer::new(&old), BufferUsage::transfer_read());
			let src_as = pass.resource(ExternalBuffer::new(&old_as), BufferUsage::transfer_read());
			let dst = pass.resource(ExternalBuffer::new(instances), BufferUsage::transfer_write());
			let dst_as = pass.resource(ExternalBuffer::new(as_instances), BufferUsage::transfer_write());
			pass.build(move |mut pass| {
				pass.copy_full_buffer(src, dst, 0);
				pass.copy_full_buffer(src_as, dst_as, 0);
			});
			frame.delete(old);
			frame.delete(old_as);

			Some((dst, dst_as))
		} else {
			None
		};
		let ls = if *light_len > *light_cap {
			while *light_len > *light_cap {
				*light_cap *= 2;
			}

			let new = Buffer::create(
				frame.device(),
				resource::BufferDesc {
					name: "scene lights",
					size: *light_cap as u64 * std::mem::size_of::<GpuLight>() as u64,
					readback: false,
				},
			)
			.unwrap();
			let old = std::mem::replace(lights, new);

			let mut pass = frame.pass("copy lights");
			let src = pass.resource(ExternalBuffer::new(&old), BufferUsage::transfer_read());
			let dst = pass.resource(ExternalBuffer::new(lights), BufferUsage::transfer_write());
			pass.build(move |mut pass| pass.copy_full_buffer(src, dst, 0));
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
		let (instances, as_instances) = if let Some((instances, as_instances)) = ins {
			pass.reference(instances, usages);
			pass.reference(as_instances, usages);
			(instances, as_instances)
		} else {
			let i = pass.resource(ExternalBuffer::new(instances), usages);
			let a = pass.resource(ExternalBuffer::new(as_instances), usages);
			(i, a)
		};
		let lights = if let Some(lights) = ls {
			pass.reference(lights, usages);
			lights
		} else {
			pass.resource(ExternalBuffer::new(lights), usages)
		};
		let update_buffer = (count > 0).then(|| {
			pass.resource(
				BufferDesc::upload((count * std::mem::size_of::<GpuSceneUpdate>()) as u64),
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

				let push = PushConstants {
					instances: pass.get(instances).ptr(),
					as_instances: pass.get(as_instances).ptr(),
					lights: pass.get(lights).ptr(),
					updates: update_buf.ptr(),
					frame: frame_index,
					count,
					_pad: 0,
				};
				self.pass.dispatch(&mut pass, &push, (count + 63) / 64, 1, 1);

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
			BufferDesc::gpu(sinfo.build_scratch_size),
			BufferUsage {
				usages: &[BufferUsageType::AccelerationStructureBuildScratch],
			},
		);
		let addr = as_.addr();
		let handle = as_.buf_handle();
		let as_buf = pass.resource(
			ExternalBuffer { handle },
			BufferUsage {
				usages: &[
					BufferUsageType::AccelerationStructureBuildRead,
					BufferUsageType::AccelerationStructureBuildWrite,
				],
			},
		);
		pass.build(move |mut pass| unsafe {
			pass.device.as_ext().cmd_build_acceleration_structures(
				pass.buf,
				&[vk::AccelerationStructureBuildGeometryInfoKHR::default()
					.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
					.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
					.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
					.geometries(&geo)
					.dst_acceleration_structure(as_.handle())
					.scratch_data(vk::DeviceOrHostAddressKHR {
						device_address: pass.get(scratch).ptr::<u8>().addr(),
					})],
				&[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
					.primitive_count(count)
					.primitive_offset(0)]],
			);
		});

		SceneReader {
			instances,
			as_: as_buf,
			lights,
			light_count: *light_len,
			sun_dir: *sun_dir,
			sun_radiance: *sun_radiance,
			as_offset: addr - handle.addr,
			instance_count: *len,
			max_depth: depth_refs.first_key_value().map(|(InvertOrd(d), _)| *d).unwrap_or(0),
			frame: frame_index,
		}
	}
}

#[derive(Copy, Clone)]
pub struct SceneReader {
	pub instances: Res<BufferHandle>,
	pub as_: Res<BufferHandle>,
	pub lights: Res<BufferHandle>,
	pub light_count: u32,
	pub sun_dir: Vec3<f32>,
	pub sun_radiance: Vec3<f32>,
	pub as_offset: u64,
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

struct MatBuf {
	materials: Buffer,
	len: u32,
	cap: u32,
	map: FxHashMap<AWeak<Material>, GpuPtr<GpuMaterial>>,
}

impl MatBuf {
	pub fn new(device: &Device) -> Result<Self> {
		Ok(Self {
			materials: Buffer::create(
				device,
				resource::BufferDesc {
					name: "scene materials",
					size: std::mem::size_of::<GpuMaterial>() as u64 * 4096,
					readback: false,
				},
			)?,
			len: 0,
			cap: 4096,
			map: FxHashMap::default(),
		})
	}

	fn id(img: &Option<ARef<Image>>) -> Option<ImageId> { img.as_ref().map(|x| x.view().id.unwrap()) }

	// TODO: remove and resize materials
	fn get_material(&mut self, mat: &ARef<Material>) -> GpuPtr<GpuMaterial> {
		*self.map.entry(mat.downgrade()).or_insert_with(|| {
			let i = self.len;
			self.len += 1;
			if i == self.cap {
				panic!("too many materials");
			}
			cast_slice_mut::<_, GpuMaterial>(unsafe { self.materials.data().as_mut() })[i as usize] = GpuMaterial {
				base_color: Self::id(&mat.base_color),
				base_color_factor: mat.base_color_factor,
				metallic_roughness: Self::id(&mat.metallic_roughness),
				metallic_factor: mat.metallic_factor,
				roughness_factor: mat.roughness_factor,
				normal: Self::id(&mat.normal),
				emissive: Self::id(&mat.emissive),
				emissive_factor: mat.emissive_factor,
			};
			self.materials.ptr().offset(i as _)
		})
	}
}

pub struct Scene {
	instances: Buffer,
	as_instances: Buffer,
	lights: Buffer,
	materials: MatBuf,
	as_: AS,
	len: u32,
	cap: u32,
	light_len: u32,
	light_cap: u32,
	entity_map: FxHashMap<Entity, Vec<(u32, u32)>>,
	light_map: FxHashMap<Entity, u32>,
	updates: Vec<GpuSceneUpdate>,
	depth_refs: BTreeMap<InvertOrd<u32>, u32>,
	sun_dir: Vec3<f32>,
	sun_radiance: Vec3<f32>,
}

fn map_aabb(aabb: Aabb<f32>) -> GpuAabb {
	GpuAabb {
		center: aabb.center(),
		half_extent: aabb.half_size().into(),
	}
}

pub(crate) fn map_transform(transform: &Transform) -> GpuTransform {
	GpuTransform {
		position: transform.position,
		rotation: transform.rotation,
		scale: transform.scale,
	}
}

fn map_light_ty(ty: LightType) -> GpuLightType {
	match ty {
		LightType::Point => GpuLightType::Point,
		LightType::Directional => GpuLightType::Directional,
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
					size: std::mem::size_of::<GpuInstance>() as u64 * 4096,
					readback: false,
				},
			)?,
			as_instances: Buffer::create(
				device,
				resource::BufferDesc {
					name: "AS scene instances",
					size: std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64 * 4096,
					readback: false,
				},
			)?,
			lights: Buffer::create(
				device,
				resource::BufferDesc {
					name: "scene lights",
					size: std::mem::size_of::<GpuLight>() as u64 * 4096,
					readback: false,
				},
			)?,
			materials: MatBuf::new(device)?,
			as_: AS::default(),
			len: 0,
			cap: 4096,
			light_len: 0,
			light_cap: 4096,
			entity_map: FxHashMap::default(),
			light_map: FxHashMap::default(),
			updates: Vec::new(),
			depth_refs: BTreeMap::new(),
			sun_dir: Vec3::new(0.0, 1.0, -0.005).normalized(),
			sun_radiance: Vec3::broadcast(20.0),
		})
	}

	pub fn add(&mut self, entity: Entity, transform: &Transform, mesh: &MeshComponent) {
		let data = mesh
			.inner
			.iter()
			.map(|m| {
				let instance = self.len;
				self.len += 1;
				let mat = m.material();
				let material = self.materials.get_material(mat);
				self.updates.push(GpuSceneUpdate::add(instance, transform, m, material));
				if mat.emissive_factor != Vec3::zero() {
					self.updates
						.push(GpuSceneUpdate::add_emissive(self.light_len, instance));
					self.light_map.insert(entity, self.light_len);
					self.light_len += 1;
				}

				let depth = m.bvh_depth();
				*self.depth_refs.entry(InvertOrd(depth)).or_insert(0) += 1;
				(instance, depth)
			})
			.collect();

		self.entity_map.insert(entity, data);
	}

	pub fn add_light(&mut self, entity: Entity, transform: &Transform, light: &LightComponent) {
		let instance = self.light_len;
		self.light_len += 1;
		self.light_map.insert(entity, instance);
		self.updates.push(GpuSceneUpdate::add_light(instance, transform, light));
		if matches!(light.ty, LightType::Directional) {
			self.sun_dir = transform.rotation * -Vec3::unit_z();
			self.sun_radiance = light.radiance;
		}
	}

	pub fn remove(&mut self, entity: Entity) {
		unimplemented!();
		for (instance, depth) in self.entity_map.remove(&entity).expect("entity not in scene") {
			let depth = self.depth_refs.get_mut(&InvertOrd(depth)).unwrap();
			*depth -= 1;
			if *depth == 0 {
				let d = *depth;
				self.depth_refs.remove(&InvertOrd(d));
			}

			// TODO: deal with lights and materials.
			self.updates.push(GpuSceneUpdate::remove(instance, self.len - 1));
			self.len -= 1;
		}
	}

	pub fn remove_light(&mut self, entity: Entity) {
		let instance = self.light_map.remove(&entity).expect("entity not in scene");
		self.updates
			.push(GpuSceneUpdate::remove_light(instance, self.light_len - 1));
		self.light_len -= 1;
	}

	pub fn change_mesh_and_transform(&mut self, entity: Entity, transform: &Transform, mesh: &MeshComponent) {
		for ((instance, depth), m) in self
			.entity_map
			.get_mut(&entity)
			.expect("entity not in scene")
			.iter_mut()
			.zip(&mesh.inner)
		{
			let old_depth = self.depth_refs.get_mut(&InvertOrd(*depth)).unwrap();
			*old_depth -= 1;
			if *old_depth == 0 {
				self.depth_refs.remove(&InvertOrd(*depth));
			}

			// TODO: deal with lights.
			let material = self.materials.get_material(m.material());
			self.updates
				.push(GpuSceneUpdate::add(*instance, transform, m, material));

			let new_depth = m.bvh_depth();
			*self.depth_refs.entry(InvertOrd(new_depth)).or_insert(0) += 1;
			*depth = new_depth;
		}
	}

	pub fn change_mesh(&mut self, entity: Entity, mesh: &MeshComponent) {
		for ((instance, depth), m) in self
			.entity_map
			.get_mut(&entity)
			.expect("entity not in scene")
			.iter_mut()
			.zip(&mesh.inner)
		{
			let old_depth = self.depth_refs.get_mut(&InvertOrd(*depth)).unwrap();
			*old_depth -= 1;
			if *old_depth == 0 {
				self.depth_refs.remove(&InvertOrd(*depth));
			}

			// TODO: deal with lights.
			let material = self.materials.get_material(m.material());
			self.updates.push(GpuSceneUpdate::change_mesh(*instance, m, material));

			let new_depth = m.bvh_depth();
			*self.depth_refs.entry(InvertOrd(new_depth)).or_insert(0) += 1;
			*depth = new_depth;
		}
	}

	pub fn change_transform(&mut self, entity: Entity, transform: &Transform) {
		for (instance, _) in self.entity_map.get(&entity).expect("entity not in scene") {
			self.updates
				.push(GpuSceneUpdate::change_transform(*instance, transform));
		}
	}
}
