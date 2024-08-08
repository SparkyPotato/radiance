use bytemuck::{AnyBitPattern, NoUninit};
use crossbeam_channel::Sender;
use radiance_asset::{Asset, AssetSource};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Vec3, Vec4};

use crate::asset::{
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	Loader,
};

#[derive(Copy, Clone, NoUninit, AnyBitPattern)]
#[repr(C)]
pub struct GpuMaterial {
	pub base_color_factor: Vec4<f32>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	pub emissive_factor: Vec3<f32>,
}

const_assert_eq!(std::mem::size_of::<GpuMaterial>(), 36);
const_assert_eq!(std::mem::align_of::<GpuMaterial>(), 4);

pub struct Material {
	pub(super) index: u32,
}

impl RuntimeAsset for Material {
	fn into_resources(self, queue: Sender<DelRes>) { queue.send(DelRes::Material(self.index)).unwrap(); }
}

impl<S: AssetSource> Loader<'_, S> {
	pub fn load_material(&mut self, uuid: Uuid) -> LResult<Material, S> {
		match AssetRuntime::get_cache(&mut self.runtime.materials, uuid) {
			Some(x) => Ok(x),
			None => {
				let m = self.load_material_from_disk(uuid)?;
				self.runtime.materials.insert(uuid, m.downgrade());
				Ok(m)
			},
		}
	}

	fn load_material_from_disk(&mut self, material: Uuid) -> LResult<Material, S> {
		let Asset::Material(m) = self.sys.load(material)? else {
			unreachable!("Material asset is not a material");
		};

		let mat = GpuMaterial {
			base_color_factor: m.base_color_factor,
			metallic_factor: m.metallic_factor,
			roughness_factor: m.roughness_factor,
			emissive_factor: m.emissive_factor,
		};

		// TODO: Write data
		let index = 0;
		Ok(RRef::new(Material { index }, self.runtime.deleter.clone()))
	}
}
