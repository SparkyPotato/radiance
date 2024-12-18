implementing graph;

public struct AS {
	u64 addr;

	public RaytracingAccelerationStructure get() {
		return spirv_asm {
			OpExtension "SPV_KHR_ray_tracing";
			OpCapability RayTracingKHR;
			result: $$RaytracingAccelerationStructure = OpConvertUToAccelerationStructureKHR $addr;
		};
	}
}

