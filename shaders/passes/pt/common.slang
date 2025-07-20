module common;

import graph;
import graph.util.rng;
import asset;
import asset.light;
import passes.bsdf;
import passes.sky;

public struct PushConstants {
	public RtInstance<NonUniform>* instances;
	public LightTree lights;
	public Camera* camera;
	public AS as;
	public Sampler sampler;
	public STex2D<f32x4, rgba32f> output;
	public Tex2D<f32> ggx_energy_compensation_lut;
	public Rng rng;
	public u32 samples;
	public SkySampler sky;
}

[vk::push_constant]
public PushConstants Constants;

public struct Ray {
	public f32x3 origin;
	public f32x3 dir;
	public f32 t;

	public __init() {
		this.origin = f32x3(0.f);
		this.dir = f32x3(0.f);
		this.t = 0.f;
	}

	public __init(f32x3 origin, f32x3 dir, f32 t = 1e10f) {
		this.origin = origin;
		this.dir = dir;
		this.t = t;
	}

	public void trace<T>(u32 flags, u32 miss, inout T payload) {
		TraceRay(Constants.as.get(), flags, 0xff, 0, 0, miss, this.to_rt(), payload);
	}

	RayDesc to_rt() {
		RayDesc r;
		r.Origin = this.origin;
		r.Direction = this.dir;
		r.TMin = 1e-5f;
		r.TMax = this.t;
		return r;
	}
}

public struct HitPayload {
	public Rng rng;
	public Ray ray;
	public f32x3 L;
	public f32 p_bounce;
	public bool specular;
	public f32x3 b;
	public f32x3 prev_hit_base_color;
	public f32x3 prev_hit_norm;
	// chit/miss -> rgen
	public bool hit;
}
