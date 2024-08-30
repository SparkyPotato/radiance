implementing graph;

[vk::binding(2, 0)]
SamplerState Samplers[];

public struct Sampler<U : Uniformity = Uniform> {
  u32 index;

  public SamplerState get() { return Samplers[U.wrap_index(this.index)]; }
}
