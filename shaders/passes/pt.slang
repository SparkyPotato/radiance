module pt;

import graph;
import graph.util;
import graph.util.rng;
import asset;
import passes.brdf;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	STex2D<f32x3> output;
	Rng rng;
	AS as;
	u32 samples;
};

[vk::push_constant]
PushConstants Constants;

struct Ray {
	f32x3 origin;
	f32x3 dir;
	f32 t;

	__init(f32x3 origin, f32x3 dir, f32 t = 1e10f) {
		this.origin = origin;
		this.dir = dir;
		this.t = t;
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

Ray primary_ray(inout Rng rng, u32x2 pix) {
	let size = Constants.output.size();
	let uv = (f32x2(pix) + rng.sample2()) / f32x2(size);
	let clip = f32x2(uv.x, uv.y) * 2.f - 1.f;
	let cam = *Constants.camera;
	let origin = mul(cam.inv_view(), f32x4(0.f, 0.f, 0.f, 1.f)).xyz;
	let view_dir = normalize(mul(cam.inv_proj(), f32x4(clip.x, -clip.y, 1.f, 1.f)).xyz);
	let dir = mul(cam.inv_view(), f32x4(view_dir, 0.f)).xyz;
	return Ray(origin, dir);
}

struct Hit {
	f32x3 position;
	f32x3 normal;
	f32x3 emissive;
	f32x3x3 to_shading_basis;
	f32x3x3 from_shading_basis;
	ShadingParams params;

	f32x3 ray_origin() {
		return this.position + this.normal * 1e-5f;
	}

	f32x3 to_shading(f32x3 x) {
		return mul(this.to_shading_basis, x);
	}

	f32x3 from_shading(f32x3 x) {
		return mul(this.from_shading_basis, x);
	}
}

Optional<Hit> trace_ray(Ray ray) {
	RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, ray.to_rt());
	while (q.Proceed());
	if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
		Hit h;

		let instance = Constants.instances[q.CommittedRayInstanceCustomIndex()];
		let tri = q.CommittedRayPrimitiveIndex();
		let iptr = (u32*)(instance.raw_mesh + instance.raw_vertex_count);
		let indices = u32x3(iptr[tri * 3], iptr[tri * 3 + 1], iptr[tri * 3 + 2]);
		let v0 = instance.raw_mesh[indices.x];
		let v1 = instance.raw_mesh[indices.y];
		let v2 = instance.raw_mesh[indices.z];
		let b = q.CommittedRayBarycentrics();
		let bary = f32x3(1.f - b.x - b.y, b.x, b.y);
		let mat = instance.material;

		let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
		let tmat = instance.transform.mat();
		let position = mul(tmat, f32x4(bary.x * v0.position + bary.y * v1.position + bary.z * v2.position, 1.f)).xyz;
		let normal = normalize(mul(tmat, f32x4(bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal, 0.f)).xyz);

		let bc = mat->base_color.get();
		let mr = mat->metallic_roughness.get();
		let nm = mat->normal.get();
		let em = mat->emissive.get();
		let white = f32x4(1.f);
		let base_color = (bc.load(bc.pixel_of_uv(uv), white) * mat->base_color_factor).xyz;
		let met_rough = mr.load(mr.pixel_of_uv(uv), white).xy;
		let metallic = met_rough.x * mat->metallic_factor;
		let roughness = met_rough.y * mat->roughness_factor;

		h.position = position;
		h.normal = f32x3(0.f, 0.f, 1.f);
		h.emissive = f32x3(0.f);
		h.from_shading_basis = gen_basis(normal);
		h.to_shading_basis = transpose(h.from_shading_basis);
		h.params.base_color = base_color;
		h.params.roughness = roughness;
		h.params.metallic = metallic;
		h.params.rel_ior = 1.5f;
		h.params.ior = 1.5f;
		return h;
	} else
		return none;
}

struct LightSample {
	f32x3 L;
	f32x3 wi;
	f32 t;
	f32 pdf;
}

f32x3 sample_sky() {
	return f32x3(0.43f, 0.61f, 0.92f) * 2.f;
}

LightSample sample_light(inout Rng rng, Hit hit) {
	let L = sample_sky();
	let wi = rng.sample_cos_hemi();
	let pdf = wi.z / PI;
	return { L, wi, 1e10f, pdf };
}

bool is_unshadowed(Hit hit, LightSample ls) {
	let ray = Ray(hit.ray_origin(), hit.from_shading(ls.wi), ls.t);
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, ray.to_rt());
	q.Proceed();
	return q.CommittedStatus() == COMMITTED_NOTHING;
}

f32x3 estimate_with_light_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let ls = sample_light(rng, hit);
	if (any(ls.L <= 0.f) || ls.pdf <= 0.f)
		return 0.f;
	let f = eval_disney_bsdf(hit.params, wo, ls.wi, false).L;
	if (any(f <= 0.f))
		return 0.f;
	let unshadowed = is_unshadowed(hit, ls);
	return f32(unshadowed) * f * ls.L / ls.pdf;
}

struct BsdfEstimate {
	f32x3 b;
	Ray r;
}

BsdfEstimate estimate_with_bsdf_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let bsdf = sample_disney_bsdf(rng, hit.params, wo, false);
	let b = bsdf.L * abs(dot(bsdf.wi, hit.normal));
	let r = Ray(hit.ray_origin(), hit.from_shading(bsdf.wi));
	return { b, r };
}

f32x3 li(inout Rng rng, Ray ray) {
	f32x3 L = f32x3(0.f);
	f32x3 b = f32x3(1.f);
	var specular = true;

	for (int i = 0; i < 10; i++) {
		let h = trace_ray(ray);
		if (!h.hasValue) {
			if (specular)
				L += b * sample_sky();
			break;
		}
		let hit = h.value;

		if (specular)
			L += b * hit.emissive;

		let wo = hit.to_shading(-ray.dir);
		L += b * estimate_with_light_sample(rng, hit, wo);
		let es = estimate_with_bsdf_sample(rng, hit, wo);
		b *= es.b;
		ray = es.r;
	}

	return L;
}

[shader("pixel")]
void main(ScreenOutput input) {
	let pix = Constants.output.pixel_of_uv(input.uv);
	var rng = Constants.rng.init_at(pix);

	let r = primary_ray(rng, pix);
	f32x3 ret = li(rng, r);

	let prev = Constants.output.load(pix);
	let n = f32(Constants.samples);
	Constants.output.store(pix, lerp(prev, ret, 1.f / (n + 1.f)));
}

f32x3x3 gen_basis(f32x3 z) {
	f32x3 other;
	if (z.x < 0.1f && z.y < 0.1f) {
		other = f32x3(0.f, -z.z, z.y);
	} else {
		other = f32x3(-z.y, z.x, 0.f);
	}
	other = normalize(other);
	let last = cross(other, z);
	// clang-format off
	return { 
		other.x, last.x, z.x,
		other.y, last.y, z.y,
		other.z, last.z, z.z
	};
	// clang-format on
}
