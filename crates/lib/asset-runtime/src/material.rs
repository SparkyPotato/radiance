use bytemuck::{bytes_of, NoUninit};
use crossbeam_channel::Sender;
use radiance_asset::{Asset, AssetSource};
use radiance_graph::device::descriptor::ImageId;
use radiance_util::{buffer::BufSpan, staging::StageError};
use static_assertions::const_assert_eq;
use uuid::Uuid;
use vek::{Vec3, Vec4};

use crate::{
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
	pub index: u32,
}

impl RuntimeAsset for Material {
	fn into_resources(self, queue: Sender<DelRes>) {
		let size = std::mem::size_of::<GpuMaterial>() as u64;
		queue
			.send(DelRes::Material(BufSpan {
				offset: self.index as u64 * size,
				size,
			}))
			.unwrap();
	}
}

impl AssetRuntime {
	pub(crate) fn load_material_from_disk<S: AssetSource>(
		&mut self, loader: &mut Loader<'_, '_, '_, S>, material: Uuid,
	) -> LResult<Material, S> {
		let Asset::Material(m) = loader.sys.load(material)? else {
			unreachable!("Material asset is not a material");
		};

		let base_color = m
			.base_color
			.map(|x| self.load_image(loader, x, true))
			.transpose()?
			.map(|x| x.view.id.unwrap());
		let metallic_roughness = m
			.metallic_roughness
			.map(|x| self.load_image(loader, x, false))
			.transpose()?
			.map(|x| x.view.id.unwrap());
		let normal = m
			.normal
			.map(|x| self.load_image(loader, x, false))
			.transpose()?
			.map(|x| x.view.id.unwrap());
		let occlusion = m
			.occlusion
			.map(|x| self.load_image(loader, x, false))
			.transpose()?
			.map(|x| x.view.id.unwrap());
		let emissive = m
			.emissive
			.map(|x| self.load_image(loader, x, false))
			.transpose()?
			.map(|x| x.view.id.unwrap());
		let mat = GpuMaterial {
			base_color_factor: m.base_color_factor,
			base_color,
			metallic_factor: m.metallic_factor,
			roughness_factor: m.roughness_factor,
			metallic_roughness,
			normal,
			occlusion,
			emissive_factor: m.emissive_factor,
			emissive,
		};

		let BufSpan { offset, .. } = self
			.material_buffer
			.alloc(loader.ctx, loader.queue, bytes_of(&mat))
			.map_err(StageError::Vulkan)?;
		let index = (offset / std::mem::size_of::<Material>() as u64) as u32;
		Ok(RRef::new(Material { index }, loader.deleter.clone()))
	}
}

