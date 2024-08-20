#pragma once

#include "types.l.hlsl"
#include "bindings.l.hlsl"

template<typename U>
SamplerState get_sampler(u32 index);
template<>
SamplerState get_sampler<Uniform>(u32 index) {
	return Samplers[index];
}
template<>
SamplerState get_sampler<NonUniform>(u32 index) {
	return Samplers[NonUniformResourceIndex(index)];
}

template<typename U = Uniform>
struct Sampler {
	u32 index;

	SamplerState get() {
		return get_sampler<U>(index);
	}
};
