use bytemuck::NoUninit;
use crossbeam_channel::Sender;
use radiance_asset::{Asset, AssetSource};
use radiance_graph::device::descriptor::ImageId;
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Vec3, Vec4};

use crate::asset::{
	image::Image,
	rref::{RRef, RuntimeAsset},
	AssetRuntime,
	DelRes,
	LResult,
	Loader,
};

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct GpuMaterial {
	pub base_color_factor: Vec4<f32>,
	pub base_color: Option<ImageId>,
	pub metallic_factor: f32,
	pub roughness_factor: f32,
	pub metallic_roughness: Option<ImageId>,
	pub normal: Option<ImageId>,
	pub occlusion: Option<ImageId>,
	pub emissive_factor: Vec3<f32>,
	pub emissive: Option<ImageId>,
}

const_assert_eq!(std::mem::size_of::<GpuMaterial>(), 56);
const_assert_eq!(std::mem::align_of::<GpuMaterial>(), 4);

pub struct Material {
	pub(super) index: u32,
	base_color: Option<RRef<Image>>,
	metallic_roughness: Option<RRef<Image>>,
	normal: Option<RRef<Image>>,
	occlusion: Option<RRef<Image>>,
	emissive: Option<RRef<Image>>,
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

		let base_color = m.base_color.map(|x| self.load_image(x, true)).transpose()?;
		let metallic_roughness = m.metallic_roughness.map(|x| self.load_image(x, false)).transpose()?;
		let normal = m.normal.map(|x| self.load_image(x, false)).transpose()?;
		let occlusion = m.occlusion.map(|x| self.load_image(x, false)).transpose()?;
		let emissive = m.emissive.map(|x| self.load_image(x, false)).transpose()?;
		let mat = GpuMaterial {
			base_color_factor: m.base_color_factor,
			base_color: base_color.as_ref().map(|x| x.view.id.unwrap()),
			metallic_factor: m.metallic_factor,
			roughness_factor: m.roughness_factor,
			metallic_roughness: metallic_roughness.as_ref().map(|x| x.view.id.unwrap()),
			normal: normal.as_ref().map(|x| x.view.id.unwrap()),
			occlusion: occlusion.as_ref().map(|x| x.view.id.unwrap()),
			emissive_factor: m.emissive_factor,
			emissive: emissive.as_ref().map(|x| x.view.id.unwrap()),
		};

		// TODO: Write data
		let index = 0;
		Ok(RRef::new(
			Material {
				index,
				base_color,
				metallic_roughness,
				normal,
				occlusion,
				emissive,
			},
			self.runtime.deleter.clone(),
		))
	}
}
