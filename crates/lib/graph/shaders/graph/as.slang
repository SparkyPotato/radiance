module as;

import types;

[vk::binding(3, 0)] RaytracingAccelerationStructure ASes[];

public struct AS<U : Uniformity = Uniform> {
	u32 index;

	public func get() -> RaytracingAccelerationStructure {
		return ASes[U.wrap_index(this.index)];
	}

	public func trace<T>(u32 flags, u32 mask, u32 sbt_offset, u32 sbt_stride, u32 miss_group, RayDesc ray, inout T payload) {
		TraceRay<T>(
			this.get(),
			flags,
			mask,
			sbt_offset,
			sbt_stride,
			miss_group,
			ray,
			payload
		);
	}
}
