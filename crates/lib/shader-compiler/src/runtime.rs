use std::ffi::CStr;

use ash::vk;
pub use radiance_shader_compiler_macros::shader;
use rustc_hash::FxHashMap;

#[macro_export]
macro_rules! c_str {
	($name:literal) => {
		unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($name, "\0").as_bytes()) }
	};
}

pub struct ShaderBlob {
	modules: &'static [(&'static str, &'static [u8])],
}

impl ShaderBlob {
	/// Create a static shader blob. Use the `shader!` macro instead, passing the module name to it.
	pub const fn new(modules: &'static [(&'static str, &'static [u8])]) -> Self { Self { modules } }
}

pub struct ShaderRuntime {
	modules: FxHashMap<&'static str, vk::ShaderModule>,
}

impl ShaderRuntime {
	pub fn new<'s>(device: &ash::Device, modules: impl IntoIterator<Item = &'s ShaderBlob>) -> Self {
		let modules = modules
			.into_iter()
			.flat_map(|x| x.modules.iter())
			.map(|(name, spirv)| {
				let module = unsafe {
					device.create_shader_module(
						&vk::ShaderModuleCreateInfo::builder().code(std::slice::from_raw_parts(
							spirv.as_ptr() as *const u32,
							spirv.len() / 4,
						)),
						None,
					)
				}
				.unwrap();
				(*name, module)
			})
			.collect();

		Self { modules }
	}

	pub fn destroy(self, device: &ash::Device) {
		for module in self.modules.values() {
			unsafe {
				device.destroy_shader_module(*module, None);
			}
		}
	}

	pub fn shader<'a>(
		&'a self, name: &'a CStr, stage: vk::ShaderStageFlags, specialization: Option<&'a vk::SpecializationInfo>,
	) -> vk::PipelineShaderStageCreateInfoBuilder {
		let name = name.to_str().expect("shader module name is not valid utf8");
		let module = self
			.modules
			.get(name)
			.unwrap_or_else(|| panic!("shader module {} not found", name));
		let info = vk::PipelineShaderStageCreateInfo::builder()
			.stage(stage)
			.module(*module)
			.name(c_str!("main"));
		if let Some(spec) = specialization {
			info.specialization_info(spec)
		} else {
			info
		}
	}
}
