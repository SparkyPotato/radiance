implementing graph;

[vk::binding(2, 0)]
SamplerState Samplers[];

public struct Sampler<U : Uniformity = Uniform> {
	u32 index;

	public __init() {
		this.index = 0;
	}

	public SamplerState get() {
		if (U.is_uniform()) {
			return Samplers[this.index];
		} else {
			return Samplers[NonUniformResourceIndex(this.index)];
		}
	}
}
