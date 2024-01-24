use crossbeam_channel::Sender;
use radiance_asset::scene;
use radiance_util::{buffer::AllocBuffer, deletion::IntoResource};

use crate::{
	mesh::Mesh,
	rref::{RRef, RuntimeAsset},
	DelRes,
};

pub struct Scene {
	instance_buffer: AllocBuffer,
	instance_count: u32,
	meshlet_pointer_buffer: AllocBuffer,
	meshlet_pointer_count: u32,
	pub cameras: Vec<scene::Camera>,
	pub meshes: Vec<RRef<Mesh>>,
}

impl RuntimeAsset for Scene {
	fn into_resources(self, queue: Sender<DelRes>) {
		queue.send(self.instance_buffer.into_resource().into()).unwrap();
		queue.send(self.meshlet_pointer_buffer.into_resource().into()).unwrap();
	}
}

impl Scene {
	pub fn meshlet_pointer_count(&self) -> u32 { self.meshlet_pointer_count }
}

