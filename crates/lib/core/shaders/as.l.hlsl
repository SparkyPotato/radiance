#pragma once

#include "types.l.hlsl"
#include "bindings.l.hlsl"

template<typename U>
struct ASBase {};

template<>
struct ASBase<Uniform> {
	u32 index;

	RaytracingAccelerationStructure get() { return ASes[this.index]; }
};

template<>
struct ASBase<NonUniform> {
	u32 index;
	
	RaytracingAccelerationStructure get() { return ASes[NonUniformResourceIndex(this.index)]; }
};

template<typename U = Uniform>
struct AS : ASBase<U> {
	template<typename T>
	void trace(u32 flags, u32 mask, u32 sbt_offset, u32 sbt_stride, u32 miss_group, RayDesc ray, inout T payload) {
		TraceRay(
			this.get(),
			mask,
			sbt_offset,
			sbt_stride,
			miss_group,
			ray,
			payload
		);
	}
};

