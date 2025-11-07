module common;

import graph;
import graph.util.color;
import graph.util.rng;

import asset;
import passes.bsdf;
import passes.sky;
import scene.light_tree;

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

	public static Ray from_api() {
		Ray r;
		r.origin = WorldRayOrigin();
		r.dir = WorldRayDirection();
		r.t = RayTCurrent();
		return r;
	}

	public __init(f32x3 origin, f32x3 dir, f32 t = 1e10f) {
		this.origin = origin;
		this.dir = dir;
		this.t = t;
	}

	public void trace<T>(inout T payload) {
		TraceRay(Constants.as.get(), RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, this.to_rt(), payload);
	}

	public bool trace_unoccluded() {
		RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
		rq.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, this.to_rt());
		rq.Proceed();
		return rq.CommittedStatus() == COMMITTED_NOTHING;
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
	public f32x3 L;
	public f32 p_bounce;
	public bool specular;
	public f32x3 b;
	public f32x3 prev_hit_norm;
	public int bounces;

	public __init(Rng rng) {
		this.rng = rng;
		this.L = f32x3(0.f);
		this.p_bounce = 0.f;
		this.specular = true;
		this.b = f32x3(1.f);
		this.prev_hit_norm = f32x3(0.f);
		this.bounces = 0;
	}

	[mutating]
	public void trace(Ray ray) {
		ray.trace(this);
	}

	[mutating]
	public void bounce(Ray ray) {
		this.bounces++;

		let q = max(0.05f, 1.f - luminance_rec2020(this.b));
		if (this.rng.sample() < q || this.bounces >= 5)
			return this.end_trace();

		this.b /= 1.f - q;
		return ray.trace(this);
	}

	public void end_trace() {
		let pix = DispatchRaysIndex().xy;
		var ret = f32x4(this.L, 1.f);
		if (any(isnan(ret) || isinf(ret)))
			ret = f32x4(1.f, 0.f, 1.f, 1.f) * 1e7f;

		let n = Constants.samples;
		if (n == 0) {
			Constants.output.store(pix, ret);
		} else {
			let old = Constants.output.load(pix);
			Constants.output.store(pix, lerp(old, ret, 1.f / (f32(n) + 1.f)));
		}
	}
}
