use std::{collections::hash_map::Entry, fmt::Debug};

use ash::vk;
use crossbeam_channel::{Receiver, Sender};
use material::GpuMaterial;
use radiance_asset::{AssetError, AssetSource, AssetSystem};
use radiance_graph::{
	device::{Device, QueueSyncs},
	graph::Frame,
	resource::{Buffer, BufferDesc, Resource},
	util::async_exec::AsyncCtx,
};
use rref::{RRef, RWeak, RuntimeAsset};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::asset::rref::DelRes;

pub mod material;
pub mod mesh;
pub mod rref;
pub mod scene;

pub struct AssetRuntime {
	deleter: Sender<DelRes>,
	delete_recv: Receiver<DelRes>,
	scenes: FxHashMap<Uuid, RWeak<scene::Scene>>,
	materials: FxHashMap<Uuid, RWeak<material::Material>>,
	meshes: FxHashMap<Uuid, RWeak<mesh::Mesh>>,
	material_buffer: Buffer,
}

impl AssetRuntime {
	pub fn new(device: &Device) -> radiance_graph::Result<Self> {
		let (send, recv) = crossbeam_channel::unbounded();
		Ok(Self {
			deleter: send,
			delete_recv: recv,
			scenes: FxHashMap::default(),
			materials: FxHashMap::default(),
			meshes: FxHashMap::default(),
			material_buffer: Buffer::create(
				device,
				BufferDesc {
					name: "materials",
					size: std::mem::size_of::<GpuMaterial>() as u64 * 1000,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
					on_cpu: false,
				},
			)?,
		})
	}

	pub unsafe fn destroy(self, device: &Device) {
		for (_, s) in self.scenes {
			assert!(
				s.upgrade().is_none(),
				"Cannot destroy `AssetRuntime` with scene still alive"
			)
		}
		for (_, m) in self.materials {
			assert!(
				m.upgrade().is_none(),
				"Cannot destroy `AssetRuntime` with materials still alive"
			)
		}
		for (_, m) in self.meshes {
			assert!(
				m.upgrade().is_none(),
				"Cannot destroy `AssetRuntime` with meshes still alive"
			)
		}

		for x in self.delete_recv.try_iter() {
			match x {
				DelRes::Resource(r) => unsafe { r.destroy(device) },
				DelRes::Material(_) => {},
			}
		}

		self.material_buffer.destroy(device);
	}

	pub fn tick(&mut self, frame: &mut Frame) {
		while let Ok(x) = self.delete_recv.try_recv() {
			match x {
				DelRes::Resource(x) => frame.delete(x),
				// TODO: delete materials
				DelRes::Material(_) => {},
			}
		}
	}

	pub fn load<S: AssetSource, R>(
		&mut self, device: &Device, sys: &AssetSystem<S>, ctx: AsyncCtx,
		exec: impl FnOnce(&mut Loader<'_, S>) -> Result<R, LoadError<S>>,
	) -> Result<(R, QueueSyncs), LoadError<S>> {
		let mut loader = Loader {
			runtime: self,
			device,
			sys,
			ctx,
		};
		let ret = exec(&mut loader)?;
		let sync = loader.ctx.finish(device).map_err(LoadError::Vulkan)?;
		Ok((ret, sync))
	}

	pub fn get_cache<T: RuntimeAsset>(map: &mut FxHashMap<Uuid, RWeak<T>>, uuid: Uuid) -> Option<RRef<T>> {
		match map.entry(uuid) {
			Entry::Occupied(o) => match o.get().upgrade() {
				Some(x) => Some(x),
				None => {
					o.remove_entry();
					None
				},
			},
			Entry::Vacant(_) => None,
		}
	}
}

pub enum LoadError<S: AssetSource> {
	Vulkan(radiance_graph::Error),
	Asset(AssetError<S>),
}

impl<S: AssetSource> From<AssetError<S>> for LoadError<S> {
	fn from(value: AssetError<S>) -> Self { Self::Asset(value) }
}

impl<S: AssetSource> Debug for LoadError<S> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Vulkan(e) => Debug::fmt(e, f),
			Self::Asset(e) => e.fmt(f),
		}
	}
}

type LResult<T, S> = Result<RRef<T>, LoadError<S>>;

pub struct Loader<'a, S> {
	runtime: &'a mut AssetRuntime,
	device: &'a Device,
	sys: &'a AssetSystem<S>,
	ctx: AsyncCtx<'a>,
}
