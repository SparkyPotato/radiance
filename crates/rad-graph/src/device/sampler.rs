use ash::vk;
use rustc_hash::FxHashMap;

use crate::device::descriptor::{Descriptors, SamplerId};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct SamplerDesc {
	pub mag_filter: vk::Filter,
	pub min_filter: vk::Filter,
	pub mipmap_mode: vk::SamplerMipmapMode,
	pub address_mode_u: vk::SamplerAddressMode,
	pub address_mode_v: vk::SamplerAddressMode,
	pub address_mode_w: vk::SamplerAddressMode,
	pub anisotropy_enable: bool,
	pub compare_enable: bool,
	pub compare_op: vk::CompareOp,
	pub reduction_mode: vk::SamplerReductionMode,
}

impl Default for SamplerDesc {
	fn default() -> Self {
		Self {
			mag_filter: vk::Filter::LINEAR,
			min_filter: vk::Filter::LINEAR,
			mipmap_mode: vk::SamplerMipmapMode::NEAREST,
			address_mode_u: vk::SamplerAddressMode::REPEAT,
			address_mode_v: vk::SamplerAddressMode::REPEAT,
			address_mode_w: vk::SamplerAddressMode::REPEAT,
			anisotropy_enable: false,
			compare_enable: false,
			compare_op: vk::CompareOp::NEVER,
			reduction_mode: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
		}
	}
}

pub(super) struct Samplers {
	caches: FxHashMap<SamplerDesc, (vk::Sampler, SamplerId)>,
}

impl Samplers {
	pub fn new() -> Self {
		Self {
			caches: FxHashMap::default(),
		}
	}

	pub fn get(&mut self, device: &ash::Device, descriptors: &Descriptors, desc: SamplerDesc) -> SamplerId {
		if let Some((_, id)) = self.caches.get(&desc) {
			return *id;
		}

		let sampler = unsafe {
			device.create_sampler(
				&vk::SamplerCreateInfo::default()
					.mag_filter(desc.mag_filter)
					.min_filter(desc.min_filter)
					.mipmap_mode(desc.mipmap_mode)
					.address_mode_u(desc.address_mode_u)
					.address_mode_v(desc.address_mode_v)
					.address_mode_w(desc.address_mode_w)
					.anisotropy_enable(desc.anisotropy_enable)
					.compare_enable(desc.compare_enable)
					.compare_op(desc.compare_op)
					.max_anisotropy(16.0)
					.min_lod(0.0)
					.max_lod(vk::LOD_CLAMP_NONE)
					.mip_lod_bias(0.0)
					.border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
					.unnormalized_coordinates(false)
					.push_next(&mut vk::SamplerReductionModeCreateInfo::default().reduction_mode(desc.reduction_mode)),
				None,
			)
		}
		.expect("Failed to create sampler");

		let id = descriptors.get_sampler(device, sampler);
		self.caches.insert(desc, (sampler, id));
		id
	}

	pub(super) unsafe fn cleanup(&mut self, device: &ash::Device) {
		for (_, (sampler, _)) in self.caches.drain() {
			device.destroy_sampler(sampler, None);
		}
	}
}
