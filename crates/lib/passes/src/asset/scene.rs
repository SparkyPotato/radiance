use ash::vk;
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use parry3d::{
	bounding_volume::SimdAabb,
	math::{Point, SimdReal, Vector},
	na::{SimdBool, SimdPartialOrd, SimdValue},
	partitioning::{Qbvh, QbvhDataGenerator, SimdBestFirstVisitStatus, SimdBestFirstVisitor},
	query::SimdRay,
};
use radiance_asset::{scene, util::SliceWriter, Asset, AssetSource};
use radiance_graph::{
	device::{
		descriptor::{ASId, BufferId},
		Compute,
	},
	graph::Resource,
	resource::{ASDesc, Buffer, BufferDesc, Resource as _, AS},
	sync::{get_global_barrier, GlobalBarrier, UsageType},
};
use static_assertions::const_assert_eq;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Aabb, Mat4, Ray, Vec3, Vec4};

use crate::asset::{
	mesh::{Intersection, Mesh},
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	LoadError,
	Loader,
};

pub struct Node {
	name: String,
	transform: Mat4<f32>,
	inv_transform: Mat4<f32>,
	mesh: RRef<Mesh>,
	instance: u32,
}

pub struct Scene {
	instance_buffer: Buffer,
	meshlet_pointer_buffer: Buffer,
	meshlet_pointer_count: u32,
	acceleration_structure: AS,
	pub cameras: Vec<scene::Camera>,
	nodes: Vec<Node>,
	// bvh: Qbvh<u32>,
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
}

const_assert_eq!(std::mem::size_of::<GpuInstance>(), 52);
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

impl Scene {
	pub fn instances(&self) -> BufferId { self.instance_buffer.id().unwrap() }

	pub fn meshlet_pointers(&self) -> BufferId { self.meshlet_pointer_buffer.id().unwrap() }

	pub fn meshlet_pointer_count(&self) -> u32 { self.meshlet_pointer_count }

	pub fn acceleration_structure(&self) -> ASId { self.acceleration_structure.id.unwrap() }

	pub fn intersect(&self, _: Ray<f32>, _: f32) -> Option<Intersection> {
		// let s = span!(Level::TRACE, "scene intersect");
		// let _e = s.enter();
		//
		// struct Visitor<'a> {
		// 	ray: Ray<f32>,
		// 	tmax: f32,
		// 	scene: &'a Scene,
		// }
		// impl<'a> SimdBestFirstVisitor<u32, SimdAabb> for Visitor<'a> {
		// 	type Result = Intersection;
		//
		// 	fn visit(
		// 		&mut self, tmax: f32, bv: &SimdAabb, data: Option<[Option<&u32>; 4]>,
		// 	) -> SimdBestFirstVisitStatus<Self::Result> {
		// 		let ray = self.ray;
		// 		let ray = parry3d::query::Ray {
		// 			origin: Point::new(ray.origin.x, ray.origin.y, ray.origin.z),
		// 			dir: Vector::new(ray.direction.x, ray.direction.y, ray.direction.z),
		// 		};
		// 		let (hit, t) = bv.cast_local_ray(&SimdRay::splat(ray), SimdReal::splat(self.tmax));
		// 		if let Some(data) = data {
		// 			let mut ts = [0.0; 4];
		// 			let mut mask = [false; 4];
		// 			let mut res = [None; 4];
		//
		// 			let closer = t.simd_lt(SimdReal::splat(tmax));
		// 			let bitmask = (closer & hit).bitmask();
		//
		// 			for ii in 0..4 {
		// 				if (bitmask & (1 << ii)) != 0 {
		// 					if let Some(&node) = data[ii] {
		// 						let node = &self.scene.nodes[node as usize];
		// 						let transform = &node.inv_transform;
		// 						let origin = *transform * self.ray.origin.with_w(1.0);
		// 						let dir = *transform * self.ray.direction.with_w(0.0);
		// 						let ray = Ray {
		// 							origin: origin.xyz(),
		// 							direction: dir.xyz().normalized(),
		// 						};
		// 						if let Some(int) = node.mesh.intersect(ray, tmax) {
		// 							res[ii] = Some(int);
		// 							mask[ii] = true;
		// 							ts[ii] = int.t;
		// 						}
		// 					}
		// 				}
		// 			}
		//
		// 			SimdBestFirstVisitStatus::MaybeContinue {
		// 				weights: ts.into(),
		// 				mask: mask.into(),
		// 				results: res,
		// 			}
		// 		} else {
		// 			SimdBestFirstVisitStatus::MaybeContinue {
		// 				weights: t,
		// 				mask: hit,
		// 				results: [None; 4],
		// 			}
		// 		}
		// 	}
		// }
		// self.bvh
		// 	.traverse_best_first(&mut Visitor { ray, tmax, scene: self })
		// 	.map(|(_, i)| i)
		None
	}
}

impl RuntimeAsset for Scene {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(Resource::Buffer(self.instance_buffer).into()).unwrap();
		queue
			.send(Resource::Buffer(self.meshlet_pointer_buffer).into())
			.unwrap();
		queue.send(Resource::AS(self.acceleration_structure).into()).unwrap();
	}
}

impl<S: AssetSource> Loader<'_, S> {
	pub fn load_scene(&mut self, uuid: Uuid) -> LResult<Scene, S> {
		match AssetRuntime::get_cache(&mut self.runtime.scenes, uuid) {
			Some(x) => Ok(x),
			None => {
				let s = self.load_scene_from_disk(uuid)?;
				self.runtime.scenes.insert(uuid, s.downgrade());
				Ok(s)
			},
		}
	}

	fn load_scene_from_disk(&mut self, scene: Uuid) -> LResult<Scene, S> {
		let s = span!(Level::TRACE, "load scene from disk");
		let _e = s.enter();

		let Asset::Scene(s) = self.sys.load(scene)? else {
			unreachable!("Scene asset is not a scene");
		};

		let name = self
			.sys
			.human_name(scene)
			.to_owned()
			.unwrap_or("unnamed scene".to_string());
		let size = (std::mem::size_of::<GpuInstance>() * s.nodes.len()) as u64;
		let instance_buffer = Buffer::create(
			self.device,
			BufferDesc {
				name: &format!("{name} instance buffer"),
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				on_cpu: false,
			},
		)
		.map_err(LoadError::Vulkan)?;
		// instance_buffer
		// 	.alloc_size(loader.ctx, loader.queue, size)
		// 	.map_err(LoadError::Vulkan)?;
		let mut writer = SliceWriter::new(unsafe { instance_buffer.data().as_mut() });

		// let temp_build_buffer = Buffer::create(
		// 	self.device,
		// 	BufferDesc {
		// 		name: "test",
		// 		size: (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * s.nodes.len()) as u64,
		// 		usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
		// 		on_cpu: false,
		// 	},
		// )
		// .map_err(LoadError::Vulkan)?;
		// let mut awriter = SliceWriter::new(unsafe { temp_build_buffer.data().as_mut() });

		let nodes: Vec<_> = s
			.nodes
			.into_iter()
			.enumerate()
			.map(|(i, n)| {
				let mesh = self.load_mesh(n.model)?;
				writer
					.write(GpuInstance {
						transform: n.transform.cols.map(|x| x.xyz()),
						mesh: mesh.buffer.id().unwrap(),
					})
					.unwrap();
				// awriter
				// 	.write(VkAccelerationStructureInstanceKHR {
				// 		transform: vk::TransformMatrixKHR {
				// 			matrix: unsafe {
				// 				std::mem::transmute(
				// 					n.transform.transposed().cols.xyz().map(|x| x.into_array()).into_array(),
				// 				)
				// 			},
				// 		},
				// 		instance_custom_index_and_mask: vk::Packed24_8::new(i as _, 0xff),
				// 		instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
				// 		acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
				// 			device_handle: unsafe {
				// 				self.device.as_ext().get_acceleration_structure_device_address(
				// 					&vk::AccelerationStructureDeviceAddressInfoKHR::default()
				// 						.acceleration_structure(mesh.acceleration_structure.handle()),
				// 				)
				// 			},
				// 		},
				// 	})
				// 	.unwrap();
				Ok(Node {
					name: n.name,
					transform: n.transform,
					inv_transform: n.transform.inverted(),
					mesh,
					instance: i as u32,
				})
			})
			.collect::<Result<_, LoadError<S>>>()?;

		let meshlet_pointer_count: u32 = nodes.iter().map(|n| n.mesh.meshlet_count).sum();
		let size = std::mem::size_of::<GpuMeshletPointer>() as u64 * meshlet_pointer_count as u64;
		let meshlet_pointer_buffer = Buffer::create(
			self.device,
			BufferDesc {
				name: &format!("{name} meshlet pointers"),
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				on_cpu: false,
			},
		)
		.map_err(LoadError::Vulkan)?;
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
			let ext = self.device.as_ext();

			// let geo = [vk::AccelerationStructureGeometryKHR::default()
			// 	.geometry_type(vk::GeometryTypeKHR::INSTANCES)
			// 	.geometry(vk::AccelerationStructureGeometryDataKHR {
			// 		instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
			// 			.array_of_pointers(false)
			// 			.data(vk::DeviceOrHostAddressConstKHR {
			// 				device_address: temp_build_buffer.addr(),
			// 			}),
			// 	})
			// 	.flags(vk::GeometryFlagsKHR::OPAQUE)];
			let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
				.ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
				.flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
				.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
				// .geometries(&geo);
				.geometries(&[]);

			let count = nodes.len() as u32;
			let mut size = Default::default();
			ext.get_acceleration_structure_build_sizes(
				vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&info,
				// &[count],
				&[],
				&mut size,
			);

			let as_ = AS::create(
				self.device,
				ASDesc {
					name: &format!("{name} AS"),
					flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
					ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
					size: size.acceleration_structure_size,
				},
			)
			.map_err(LoadError::Vulkan)?;

			// let scratch = Buffer::create(
			// 	self.device,
			// 	BufferDesc {
			// 		name: "AS build scratch buffer",
			// 		size: size.build_scratch_size,
			// 		usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			// 		on_cpu: false,
			// 	},
			// )
			// .map_err(LoadError::Vulkan)?;

			// info.dst_acceleration_structure = as_.handle();
			// info.scratch_data = vk::DeviceOrHostAddressKHR {
			// 	device_address: scratch.addr(),
			// };

			// let buf = self.ctx.get_buf::<Compute>();

			// self.device.device().cmd_pipeline_barrier2(
			// 	buf,
			// 	&vk::DependencyInfo::default().memory_barriers(&[get_global_barrier(&GlobalBarrier {
			// 		previous_usages: &[UsageType::AccelerationStructureBuildWrite],
			// 		next_usages: &[UsageType::AccelerationStructureBuildRead],
			// 	})]),
			// );
			// ext.cmd_build_acceleration_structures(
			// 	buf,
			// 	&[info],
			// 	&[&[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)]],
			// );
			//
			// self.ctx.delete::<Compute>(temp_build_buffer);
			// self.ctx.delete::<Compute>(scratch);

			as_
		};

		// struct Gen<'a> {
		// 	nodes: &'a Vec<Node>,
		// }
		// impl QbvhDataGenerator<u32> for Gen<'_> {
		// 	fn size_hint(&self) -> usize { self.nodes.len() as _ }
		//
		// 	fn for_each(&mut self, mut f: impl FnMut(u32, parry3d::bounding_volume::Aabb)) {
		// 		for (i, n) in self.nodes.iter().enumerate() {
		// 			let parry3d::bounding_volume::Aabb { mins, maxs } = n.mesh.mesh.local_aabb();
		// 			let aabb = Aabb {
		// 				min: Vec3::new(mins.x, mins.y, mins.z),
		// 				max: Vec3::new(maxs.x, maxs.y, maxs.z),
		// 			};
		// 			let center = aabb.center();
		// 			let extent = n.transform.map(f32::abs) * (aabb.max - center).with_w(0.0);
		// 			let center = n.transform * center.with_w(1.0);
		// 			let min = center - extent;
		// 			let max = center + extent;
		// 			let aabb = parry3d::bounding_volume::Aabb {
		// 				mins: Point::new(min.x, min.y, min.z),
		// 				maxs: Point::new(max.x, max.y, max.z),
		// 			};
		// 			f(i as _, aabb)
		// 		}
		// 	}
		// }
		// let mut bvh = Qbvh::new();
		// bvh.clear_and_rebuild(Gen { nodes: &nodes }, 0.0);

		Ok(RRef::new(
			Scene {
				nodes,
				instance_buffer,
				meshlet_pointer_buffer,
				acceleration_structure,
				meshlet_pointer_count,
				cameras: s.cameras,
				// bvh,
			},
			self.runtime.deleter.clone(),
		))
	}
}
