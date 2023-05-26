use std::ffi::CStr;

use ash::vk;
use rustc_hash::FxHashMap;

#[macro_export]
macro_rules! shader {
	($name:literal) => {
		$crate::runtime::ShaderBlob::new($name, include_bytes!(env!(concat!($name, "_OUTPUT_PATH"))))
	};
}

#[macro_export]
macro_rules! c_str {
	($name:literal) => {
		unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($name, "\0").as_bytes()) }
	};
}

pub struct ShaderBlob {
	name: &'static str,
	spirv: &'static [u8],
}

impl ShaderBlob {
	/// Create a static shader blob. Use the `shader!` macro instead, passing the module name to it.
	pub const fn new(name: &'static str, spirv: &'static [u8]) -> Self { Self { name, spirv } }
}

pub struct ShaderRuntime {
	modules: FxHashMap<&'static str, vk::ShaderModule>,
}

impl ShaderRuntime {
	pub fn new(device: &ash::Device, modules: &[ShaderBlob]) -> Self {
		let modules = modules
			.iter()
			.map(|shader| {
				let module = unsafe {
					device.create_shader_module(
						&vk::ShaderModuleCreateInfo::builder().code(std::slice::from_raw_parts(
							shader.spirv.as_ptr() as *const u32,
							shader.spirv.len() / 4,
						)),
						None,
					)
				}
				.unwrap();
				(shader.name, module)
			})
			.collect();
		Self { modules }
	}

	pub unsafe fn destroy(self, device: &ash::Device) {
		for module in self.modules.values() {
			device.destroy_shader_module(*module, None);
		}
	}

	pub fn shader<'a>(
		&'a self, name: &'a CStr, stage: vk::ShaderStageFlags, specialization: Option<&'a vk::SpecializationInfo>,
	) -> vk::PipelineShaderStageCreateInfoBuilder {
		let utf8 = name.to_str().expect("shader module name is not valid utf8");
		let module = utf8.split('/').next().expect("expected shader module name");
		let module = self.modules.get(module).expect("shader module not found");
		let info = vk::PipelineShaderStageCreateInfo::builder()
			.stage(stage)
			.module(*module)
			.name(name);
		if let Some(spec) = specialization {
			info.specialization_info(spec)
		} else {
			info
		}
	}
}
