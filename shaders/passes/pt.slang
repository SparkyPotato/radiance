module pt;

import graph;
import graph.util;
import graph.util.color;
import graph.util.rng;
import asset;
import passes.bsdf;

struct PushConstants {
	Instance<NonUniform>* instances;
	Light* lights;
	Camera* camera;
	AS as;
	Sampler sampler;
	STex2D<f32x3> output;
	Rng rng;
	u32 samples;
	u32 light_count;
	u32 sky_light;
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
	let view_dir = normalize(mul(cam.inv_proj(), f32x4(clip.x, -clip.y, 0.f, 1.f)).xyz);
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
		let instance = Constants.instances[q.CommittedRayInstanceId()];
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

		let s = Constants.sampler;
		let bc = mat->base_color.get();
		let mr = mat->metallic_roughness.get();
		let nm = mat->normal.get();
		let em = mat->emissive.get();
		let white = f32x4(1.f);
		let met_rough = mr.sample(s, uv, white).xy;
		let emissive = em.sample(s, uv, white).xyz * mat->emissive_factor;
		let rough = met_rough.y * mat->roughness_factor;

		Hit h = {};
		h.position = position;
		h.normal = f32x3(0.f, 0.f, 1.f);
		h.emissive = emissive;
		h.from_shading_basis = gen_basis(normal);
		h.to_shading_basis = transpose(h.from_shading_basis);
		h.params.base_color = (bc.sample(s, uv, white) * mat->base_color_factor).xyz;
		h.params.roughness = max(rough * rough, 0.0001f);
		h.params.metallic = met_rough.x * mat->metallic_factor;
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

f32x3 sample_sky(f32x3 dir) {
	if (Constants.sky_light == -1)
		return f32x3(0.f);
	return Constants.lights[Constants.sky_light].radiance;
}

LightSample sample_light(inout Rng rng, Hit hit, Light light) {
	switch (light.ty) {
		case LightType.Point: {
			let pos = light.pos_or_dir;
			let dir = pos - hit.position;
			let t2 = dot(dir, dir);
			let L = light.radiance / t2;
			let t = sqrt(t2);
			let wi = hit.to_shading(dir / t);
			let pdf = 1.f;
			return { L, wi, t, pdf };
		}
		case LightType.Directional: {
			let L = light.radiance;
			let wi = hit.to_shading(-light.pos_or_dir);
			let pdf = 1.f;
			return { L, wi, 1e10f, pdf };
		}
		case LightType.Emissive: {
			let i = asuint(light.radiance.x);
			let instance = Constants.instances[i];

			let tri_count = instance.raw_tri_count;
			let tri = min(u32(rng.sample() * tri_count), tri_count - 1);
			let iptr = (u32*)(instance.raw_mesh + instance.raw_vertex_count);
			let indices = u32x3(iptr[tri * 3], iptr[tri * 3 + 1], iptr[tri * 3 + 2]);
			let v0 = instance.raw_mesh[indices.x];
			let v1 = instance.raw_mesh[indices.y];
			let v2 = instance.raw_mesh[indices.z];
			let b = rng.sample2();
			let su0 = sqrt(b.x);
			let u = 1.f - su0;
			let v = b.y * su0;
			let w = 1.f - u - v;

			let pos = u * v0.position + v * v1.position + w * v2.position;
			let uv = u * v0.uv + v * v1.uv + w * v2.uv;
			let area = length(cross(v1.position - v0.position, v2.position - v0.position)) * 0.5f;

			let mat = instance.material;
			let em = mat->emissive.get();
			let emf = mat->emissive_factor;

			let dir = pos - hit.position;
			let t2 = dot(dir, dir);
			let t = sqrt(t2);
			let wi = hit.to_shading(dir / t);
			let L = em.load(em.pixel_of_uv(uv), f32x4(1.f)).xyz * emf / t2;

			return { L, wi, t, 1.f / (area * f32(tri_count)) };
		}
		case LightType.Sky: {
			let L = light.radiance;
			let wi = rng.sample_cos_hemi();
			let pdf = wi.z / PI;
			return { L, wi, 1e10f, pdf };
		}
	}

	return { f32x3(0.f), f32x3(0.f), 0.f, 0.f };
}

LightSample sample_one_light(inout Rng rng, Hit hit) {
	let n = f32(Constants.light_count);
	if (n == 0)
		return { f32x3(0.f), f32x3(0.f), 0.f, 0.f };

	let l = min(u32(rng.sample() * n), u32(n - 1.f));
	let light = Constants.lights[l];
	var ls = sample_light(rng, hit, light);
	ls.pdf /= n;
	return ls;
}

bool is_unshadowed(Hit hit, LightSample ls) {
	let ray = Ray(hit.ray_origin(), hit.from_shading(ls.wi), ls.t);
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, ray.to_rt());
	q.Proceed();
	return q.CommittedStatus() == COMMITTED_NOTHING;
}

f32x3 estimate_with_light_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let ls = sample_one_light(rng, hit);
	if (any(ls.L <= 0.f) || ls.pdf <= 0.f)
		return 0.f;
	let f = eval_bsdf(hit.params, wo, ls.wi);
	if (any(f <= 0.f))
		return 0.f;
	let unshadowed = is_unshadowed(hit, ls);
	return f32(unshadowed) * f * ls.L / ls.pdf;
}

struct BsdfEstimate {
	f32x3 b;
	Ray r;
	bool spec;
}

BsdfEstimate estimate_with_bsdf_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let bsdf = sample_bsdf(rng, hit.params, wo);
	let b = bsdf.L;
	let r = Ray(hit.ray_origin(), hit.from_shading(bsdf.wi));
	return { b, r, bsdf.is_specular };
}

f32x3 li(inout Rng rng, Ray ray) {
	f32x3 L = f32x3(0.f);
	f32x3 b = f32x3(1.f);
	var specular = true;

	for (int bounces = 0; bounces < 10; bounces++) {
		let h = trace_ray(ray);
		if (!h.hasValue) {
			if (specular)
				L += b * sample_sky(ray.dir);
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
		specular = es.spec;

		if (bounces > 1) {
			let q = max(0.05f, 1.f - luminance(b));
			if (rng.sample() < q)
				break;
			b /= 1.f - q;
		}
	}

	return L;
}

[shader("pixel")]
void main(ScreenOutput input) {
	let pix = Constants.output.pixel_of_uv(input.uv);
	var rng = Constants.rng.init_at(pix);

	let r = primary_ray(rng, pix);
	var ret = li(rng, r);
	if (any(isnan(ret) || isinf(ret)))
		ret = f32x3(1.f, 0.f, 1.f) * 1e7f * 0.f;

	let n = f32(Constants.samples);
	if (n == 0.f) {
		Constants.output.store(pix, ret);
	} else {
		let prev = Constants.output.load(pix);
		Constants.output.store(pix, lerp(prev, ret, 1.f / (n + 1.f)));
	}
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
