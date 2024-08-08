module sampler;

import types;

[vk::binding(2, 0)] SamplerState Samplers[];

public struct Sampler<U : Uniformity = Uniform> {
	u32 index;

	public func get() -> SamplerState {
		return Samplers[U.wrap_index(this.index)];
	}
}
