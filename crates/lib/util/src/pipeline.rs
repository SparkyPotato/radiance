use ash::vk;
use radiance_graph::{device::Device, Result};

pub struct PipelineCache {
	inner: vk::PipelineCache,
}

impl PipelineCache {
	pub fn new(device: &Device) -> Result<Self> {
		let cache = unsafe {
			device
				.device()
				.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)?
		};
		Ok(Self { inner: cache })
	}

	pub fn cache(&self) -> vk::PipelineCache { self.inner }

	pub fn dump(&self, device: &Device) -> Result<Vec<u8>> {
		unsafe {
			let data = device.device().get_pipeline_cache_data(self.inner)?;
			Ok(data)
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_pipeline_cache(self.inner, None);
		}
	}
}
