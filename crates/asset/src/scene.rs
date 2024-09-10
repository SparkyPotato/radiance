use ash::vk;
use bincode::{Decode, Encode};
use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use radiance_graph::{
	graph::Resource,
	resource::{Buffer, BufferDesc, GpuPtr, Resource as _},
};
use static_assertions::const_assert_eq;
use tracing::{span, Level};
use uuid::{uuid, Uuid};
use vek::{Mat4, Quaternion, Vec3};

use crate::{
	io::{SliceWriter, Writer},
	mesh::{map_aabb, GpuAabb, Mesh},
	rref::{DelRes, RRef},
	Asset,
	InitContext,
	LoadError,
};

#[derive(Copy, Clone, Encode, Decode, NoUninit)]
#[repr(C)]
pub struct Transform {
	#[bincode(with_serde)]
	pub translation: Vec3<f32>,
	#[bincode(with_serde)]
	pub rotation: Quaternion<f32>,
	#[bincode(with_serde)]
	pub scale: Vec3<f32>,
}

#[derive(Encode, Decode)]
pub struct DataNode {
	pub name: String,
	pub transform: Transform,
	#[bincode(with_serde)]
	pub mesh: Uuid,
}

#[derive(Copy, Clone, Encode, Decode)]
pub enum Projection {
	Perspective { yfov: f32, near: f32, far: Option<f32> },
	Orthographic { height: f32, near: f32, far: f32 },
}

#[derive(Clone, Encode, Decode)]
pub struct Camera {
	pub name: String,
	#[bincode(with_serde)]
	pub view: Mat4<f32>,
	pub projection: Projection,
}

#[derive(Encode, Decode)]
pub struct DataScene {
	pub nodes: Vec<DataNode>,
	pub cameras: Vec<Camera>,
}

pub struct Node {
	pub name: String,
	pub transform: Transform,
	pub mesh: RRef<Mesh>,
	pub instance: u32,
}

pub struct Scene {
	instance_buffer: Buffer,
	pub cameras: Vec<Camera>,
	nodes: RwLock<Vec<Node>>,
	max_depth: u32,
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuInstance {
	pub transform: Transform,
	/// Mesh buffer containing meshlets + meshlet data.
	pub mesh: GpuPtr<u8>,
	pub aabb: GpuAabb,
}

const_assert_eq!(std::mem::size_of::<GpuInstance>(), 72);
const_assert_eq!(std::mem::align_of::<GpuInstance>(), 8);

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
	pub fn instances(&self) -> GpuPtr<GpuInstance> { self.instance_buffer.ptr() }

	pub fn instance_count(&self) -> u32 { self.nodes.read().len() as _ }

	pub fn max_depth(&self) -> u32 { self.max_depth }

	pub fn node(&self, i: u32) -> MappedRwLockReadGuard<'_, Node> {
		RwLockReadGuard::map(self.nodes.read(), |x| &x[i as usize])
	}

	pub fn node_name(&self, i: u32) -> MappedRwLockWriteGuard<'_, String> {
		RwLockWriteGuard::map(self.nodes.write(), |x| &mut x[i as usize].name)
	}
}

impl Asset for Scene {
	type Import = DataScene;

	const MODIFIABLE: bool = true;
	const TYPE: Uuid = uuid!("c394ec13-387e-4af1-9873-fb4e399d4a52");

	fn initialize(mut ctx: InitContext<'_>) -> Result<RRef<Self>, LoadError> {
		let s = span!(Level::TRACE, "decode scene");
		let _e = s.enter();

		let s: DataScene = ctx.data.deserialize()?;
		let size = ((std::mem::size_of::<GpuInstance>() + std::mem::size_of::<u32>()) * s.nodes.len()) as u64;
		let instance_buffer = Buffer::create(
			ctx.device,
			BufferDesc {
				name: &format!("{} instances", ctx.name),
				size,
				usage: vk::BufferUsageFlags::STORAGE_BUFFER,
				readback: false,
			},
		)
		.map_err(LoadError::Vulkan)?;

		let mut writer = SliceWriter::new(unsafe { instance_buffer.data().as_mut() });

		let mut max_depth = 0;
		let nodes: Vec<_> = s
			.nodes
			.into_iter()
			.enumerate()
			.map(|(i, n)| {
				let mesh: RRef<Mesh> = ctx.sys.initialize(ctx.device, n.mesh)?;
				max_depth = max_depth.max(mesh.bvh_depth);
				writer
					.write(GpuInstance {
						transform: n.transform,
						mesh: mesh.buffer.ptr(),
						aabb: map_aabb(mesh.aabb),
					})
					.unwrap();

				Ok(Node {
					name: n.name,
					transform: n.transform,
					mesh,
					instance: i as u32,
				})
			})
			.collect::<Result<_, LoadError>>()?;

		Ok(ctx.make(Scene {
			instance_buffer,
			cameras: s.cameras,
			nodes: RwLock::new(nodes),
			max_depth,
		}))
	}

	fn write(&self, into: Writer) -> Result<(), std::io::Error> {
		into.serialize(DataScene {
			nodes: self
				.nodes
				.read()
				.iter()
				.map(|n| DataNode {
					name: n.name.clone(),
					transform: n.transform,
					mesh: n.mesh.uuid(),
				})
				.collect(),
			cameras: self.cameras.clone(),
		})
	}

	fn import(_: &str, import: Self::Import, into: Writer) -> Result<(), std::io::Error> { into.serialize(import) }

	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(Resource::Buffer(self.instance_buffer).into()).unwrap();
	}
}
