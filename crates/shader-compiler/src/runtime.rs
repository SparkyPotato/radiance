use std::ffi::CStr;

use ash::{
	vk::{
		PipelineShaderStageCreateInfo,
		PipelineShaderStageCreateInfoBuilder,
		ShaderModule,
		ShaderModuleCreateInfo,
		ShaderStageFlags,
		SpecializationInfo,
	},
	Device,
};
use rustc_hash::FxHashMap;
#[macro_export]
macro_rules! shader {
	($name:literal) => {
		$crate::runtime::ShaderBlob::new($name, include_bytes!(env!(concat!($name, "_OUTPUT_PATH"))))
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
	modules: FxHashMap<&'static str, ShaderModule>,
}

impl ShaderRuntime {
	pub fn new(device: &Device, modules: &[ShaderBlob]) -> Self {
		let modules = modules
			.iter()
			.map(|shader| {
				let module = unsafe {
					device.create_shader_module(
						&ShaderModuleCreateInfo::builder().code(std::slice::from_raw_parts(
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

	pub fn shader<'a>(
		&'a self, name: &'a CStr, stage: ShaderStageFlags, specialization: &'a SpecializationInfo,
	) -> PipelineShaderStageCreateInfoBuilder {
		let utf8 = name.to_str().expect("shader module name is not valid utf8");
		let module = utf8.split('/').next().expect("expected shader module name");
		let module = self.modules.get(module).expect("shader module not found");
		PipelineShaderStageCreateInfo::builder()
			.stage(stage)
			.module(*module)
			.name(name)
			.specialization_info(specialization)
	}
}
