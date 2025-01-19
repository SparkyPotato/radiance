use std::sync::RwLock;

use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable};
use rad_core::{
	asset::{
		aref::{ARef, AssetId, LARef},
		AssetView,
		BincodeAsset,
	},
	uuid,
	Engine,
};
use rad_graph::{
	device::descriptor::ImageId,
	resource::{Buffer, BufferDesc, GpuPtr, Resource},
};
use rad_world::Uuid;
use vek::{Vec3, Vec4};

use crate::assets::image::{ImageAsset, ImageAssetView};

#[derive(Encode, Decode)]
pub struct Material {
	#[bincode(with_serde)]
	pub base_color: Option<AssetId<ImageAsset>>,
	#[bincode(with_serde)]
	pub base_color_factor: Vec4<f32>,
	#[bincode(with_serde)]
	pub metallic_roughness: Option<AssetId<ImageAsset>>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	#[bincode(with_serde)]
	pub normal: Option<AssetId<ImageAsset>>,
	#[bincode(with_serde)]
	pub emissive: Option<AssetId<ImageAsset>>,
	#[bincode(with_serde)]
	pub emissive_factor: Vec3<f32>,
}

impl BincodeAsset for Material {
	const UUID: Uuid = uuid!("15695530-bc12-4745-9410-21d24480e8f1");
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMaterial {
	base_color: Option<ImageId>,
	base_color_factor: Vec4<f32>,
	metallic_roughness: Option<ImageId>,
	metallic_factor: f32,
	roughness_factor: f32,
	normal: Option<ImageId>,
	emissive: Option<ImageId>,
	emissive_factor: Vec3<f32>,
}

pub struct MaterialView {
	ptr: GpuPtr<GpuMaterial>,
	buf: BufRef,
	ctx: &'static MaterialBuffers,
	pub base_color: Option<LARef<ImageAssetView>>,
	pub metallic_roughness: Option<LARef<ImageAssetView>>,
	pub normal: Option<LARef<ImageAssetView>>,
	pub emissive: Option<LARef<ImageAssetView>>,
	pub emissive_factor: Vec3<f32>,
}

impl MaterialView {
	pub fn gpu_ptr(&self) -> GpuPtr<GpuMaterial> { self.ptr }
}

impl AssetView for MaterialView {
	type Base = Material;
	type Ctx = MaterialBuffers;

	fn load(ctx: &'static Self::Ctx, base: Self::Base) -> Result<Self, std::io::Error> { Ok(ctx.load(base)) }
}

impl Drop for MaterialView {
	fn drop(&mut self) { self.ctx.unload(self); }
}

#[derive(Copy, Clone)]
struct BufRef {
	buf: u32,
	id: u32,
}

pub struct MaterialBuffers {
	inner: RwLock<MaterialBuffersInner>,
}

struct MaterialBuffersInner {
	buffers: Vec<Buffer>,
	free: Vec<BufRef>,
	bump: u32,
}

impl Default for MaterialBuffers {
	fn default() -> Self {
		Self {
			inner: RwLock::new(MaterialBuffersInner {
				buffers: vec![Buffer::create(
					Engine::get().global(),
					BufferDesc {
						name: "materials",
						size: Self::BUFFER_SIZE * Self::MATERIAL_SIZE,
						readback: false,
					},
				)
				.unwrap()],
				free: Vec::new(),
				bump: 0,
			}),
		}
	}
}

impl MaterialBuffers {
	const BUFFER_SIZE: u64 = 1024;
	const MATERIAL_SIZE: u64 = std::mem::size_of::<GpuMaterial>() as u64;

	fn id(i: &Option<LARef<ImageAssetView>>) -> Option<ImageId> { i.as_ref().map(|i| i.id()) }

	fn load(&'static self, mat: Material) -> MaterialView {
		let mut inner = self.inner.write().unwrap();
		let buf = if let Some(free) = inner.free.pop() {
			free
		} else if inner.bump < Self::BUFFER_SIZE as u32 {
			let id = inner.bump;
			inner.bump += 1;
			BufRef {
				buf: inner.buffers.len() as u32 - 1,
				id,
			}
		} else {
			let buf = inner.buffers.len() as u32;
			inner.buffers.push(
				Buffer::create(
					Engine::get().global(),
					BufferDesc {
						name: "materials",
						size: Self::BUFFER_SIZE * Self::MATERIAL_SIZE,
						readback: false,
					},
				)
				.unwrap(),
			);
			inner.bump = 1;
			BufRef { buf, id: 0 }
		};

		let id = buf.id;
		let b = &inner.buffers[buf.buf as usize];
		let ptr = b.ptr::<GpuMaterial>().offset(id as _);

		// TODO: should we multithread these?
		// TODO: unwrap bad
		let base_color = mat.base_color.map(|id| ARef::loaded(id)).transpose().unwrap();
		let metallic_roughness = mat.metallic_roughness.map(|id| ARef::loaded(id)).transpose().unwrap();
		let normal = mat.normal.map(|id| ARef::loaded(id)).transpose().unwrap();
		let emissive = mat.emissive.map(|id| ARef::loaded(id)).transpose().unwrap();

		unsafe {
			b.data()
				.cast::<GpuMaterial>()
				.offset(id as _)
				.as_ptr()
				.write(GpuMaterial {
					base_color: Self::id(&base_color),
					base_color_factor: mat.base_color_factor,
					metallic_roughness: Self::id(&metallic_roughness),
					metallic_factor: mat.metallic_factor,
					roughness_factor: mat.roughness_factor,
					normal: Self::id(&normal),
					emissive: Self::id(&emissive),
					emissive_factor: mat.emissive_factor,
				});
		}

		MaterialView {
			ptr,
			buf,
			ctx: self,
			base_color,
			metallic_roughness,
			normal,
			emissive,
			emissive_factor: mat.emissive_factor,
		}
	}

	fn unload(&self, view: &MaterialView) {
		let mut inner = self.inner.write().unwrap();
		inner.free.push(view.buf);
	}
}

impl Drop for MaterialBuffersInner {
	fn drop(&mut self) {
		let dev = Engine::get().global();
		for buf in self.buffers.drain(..) {
			unsafe {
				buf.destroy(dev);
			}
		}
	}
}
