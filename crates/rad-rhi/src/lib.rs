use rad_core::{EngineBuilder, Module};
use rad_graph::{
	ash::{ext, vk},
	device::Device,
};

pub struct RhiModule;

impl Module for RhiModule {
	fn init(engine: &mut EngineBuilder) {
		engine.global(
			Device::builder()
				.device_extensions(&[
					ext::mesh_shader::NAME,
					ext::shader_image_atomic_int64::NAME,
					c"VK_KHR_shader_relaxed_extended_instruction",
				])
				.features(
					vk::PhysicalDeviceFeatures2::default()
						.features(
							vk::PhysicalDeviceFeatures::default()
								.shader_int16(true)
								.shader_int64(true)
								.fragment_stores_and_atomics(true),
						)
						.push_next(
							&mut vk::PhysicalDeviceVulkan11Features::default()
								.storage_push_constant16(true)
								.storage_buffer16_bit_access(true)
								.variable_pointers(true)
								.variable_pointers_storage_buffer(true),
						)
						.push_next(
							&mut vk::PhysicalDeviceVulkan12Features::default()
								.sampler_filter_minmax(true)
								.shader_int8(true)
								.storage_buffer8_bit_access(true)
								.storage_push_constant8(true)
								.scalar_block_layout(true),
						)
						.push_next(
							&mut vk::PhysicalDeviceVulkan13Features::default()
								.dynamic_rendering(true)
								.shader_demote_to_helper_invocation(true),
						)
						.push_next(&mut vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true))
						.push_next(
							&mut vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default()
								.shader_image_int64_atomics(true),
						)
						.push_next(
							&mut vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
								.ray_tracing_position_fetch(true),
						),
				)
				.build()
				.expect("Failed to init RHI")
				.0,
		);
	}
}
