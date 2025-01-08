module pt;

import graph;
import graph.util;
import graph.util.color;
import graph.util.rng;
import asset;
import passes.bsdf;
import passes.sky;

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
	SkySampler sky;
};

[vk::push_constant]
PushConstants Constants;

f32 pow_heuristic(u32 nf, f32 f_pdf, u32 ng, f32 g_pdf) {
	let f = f_pdf * f32(nf);
	let g = g_pdf * f32(ng);
	return f * f / (f * f + g * g);
}

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

	__generic<let F : u32> RayQuery<F> api_trace() {
		RayQuery<F> q;
		q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, this.to_rt());
		while (q.Proceed());
		return q;
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
	f32x3 g_normal;
	f32x3 emissive;
	f32x3x3 to_shading_basis;
	f32x3x3 from_shading_basis;
	ShadingParams params;
	f32 area;

	f32x3 ray_origin() {
		return this.position + this.g_normal * 1e-5f;
	}

	f32x3 to_shading(f32x3 x) {
		return mul(this.to_shading_basis, x);
	}

	f32x3 from_shading(f32x3 x) {
		return mul(this.from_shading_basis, x);
	}
}

struct Tbn {
	f32x3 tangent;
	f32x3 bitangent;
	f32x3 normal;

	__init(Vertex v0, Vertex v1, Vertex v2, f32x3 n) {
		// https://terathon.com/blog/tangent-space.html
		let q1 = v1.position - v0.position;
		let q2 = v2.position - v0.position;
		let st1 = v1.uv - v0.uv;
		let st2 = v2.uv - v0.uv;
		let det = st1.x * st2.y - st2.x * st1.y;
		if (det != 0.f) {
			let tc = (st2.y * q1 - st1.y * q2) / det;
			let bc = (st1.x * q2 - st2.x * q1) / det;
			let t = normalize(tc - dot(n, tc) * n);
			let h = (dot(cross(n, tc), bc) < 0.f) ? -1.f : 1.f;
			let b = h * cross(n, t);
			return { t, b, n };
		} else {
			f32x3 other;
			if (n.x < 0.1f && n.y < 0.1f) {
				other = f32x3(0.f, -n.z, n.y);
			} else {
				other = f32x3(-n.y, n.x, 0.f);
			}
			other = normalize(other);
			let last = cross(other, n);
			return { other, last, n };
		}
	}
}

Hit decode_hit(RayQuery<RAY_FLAG_FORCE_OPAQUE> q) {
	let instance = Constants.instances[q.CommittedRayInstanceId()];
	let tri = q.CommittedRayPrimitiveIndex();
	let iptr = (u32*)(instance.raw_mesh + instance.raw_vertex_count);
	let i0 = tri * 3;
	let v0 = instance.raw_mesh[iptr[i0]];
	let v1 = instance.raw_mesh[iptr[i0 + 1]];
	let v2 = instance.raw_mesh[iptr[i0 + 2]];
	let br = q.CommittedRayBarycentrics();
	let bary = f32x3(1.f - (br.x + br.y), br.x, br.y);
	let mat = instance.material;

	let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
	let normal_obj = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
	let tbn = Tbn(v0, v1, v2, normal_obj);
	let tmat = instance.transform.mat();
	let position = mul(tmat, f32x4(bary.x * v0.position + bary.y * v1.position + bary.z * v2.position, 1.f)).xyz;
	var inv_basis =
		f32x3x3(normalize(mul(tmat, f32x4(tbn.tangent, 0.f)).xyz), normalize(mul(tmat, f32x4(tbn.bitangent, 0.f)).xyz),
				normalize(mul(tmat, f32x4(tbn.normal, 0.f)).xyz));

	let s = Constants.sampler;
	let bc = mat->base_color.get();
	let mr = mat->metallic_roughness.get();
	let nm = mat->normal.get();
	let em = mat->emissive.get();
	let white = f32x4(1.f);
	let blue = f32x4(0.f, 0.f, 1.f, 1.f);

	let base = (bc.sample(s, uv, white) * mat->base_color_factor).xyz;
	let norm = nm.sample(s, uv, blue).xyz;

	let met_rough = mr.sample(s, uv, white);
	let emissive = em.sample(s, uv, white).xyz * mat->emissive_factor;
	let rough = met_rough.y * mat->roughness_factor;
	let met = met_rough.z * mat->metallic_factor;

	var basis = transpose(inv_basis);
	let g_normal = basis._m02_m12_m22;
	let norm_world = normalize(mul(basis, norm));
	basis._m02_m12_m22 = norm_world;
	inv_basis._m20_m21_m22 = norm_world;

	Hit h = {};
	h.position = position;
	h.g_normal = g_normal;
	h.emissive = emissive;
	h.from_shading_basis = basis;
	h.to_shading_basis = inv_basis;
	h.params.base_color = base;
	h.params.roughness = max(rough * rough, 0.0001f);
	h.params.metallic = met;
	h.area = length(cross(v1.position - v0.position, v2.position - v0.position)) * 0.5f;

	return h;
}

Optional<Hit> trace_ray(Ray ray) {
	let q = ray.api_trace<RAY_FLAG_FORCE_OPAQUE>();
	if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
		return decode_hit(q);
	else
		return none;
}

struct LightSample {
	f32x3 L;
	f32x3 wi;
	f32 t;
	f32 pdf;
	bool punctual;
}

f32 solid_angle_pdf(f32 theta) {
	return 1.f / (2.f * PI * (1.f - cos(theta)));
}

// TODO: shrample light (light tree) and triangles (by solid angle or area) better.
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
			return { L, wi, t, pdf, true };
		}
		case LightType.Directional: {
			let L = light.radiance;
			let wi = hit.to_shading(-light.pos_or_dir);
			let pdf = solid_angle_pdf(radians(0.5f));
			return { L, wi, 1e10f, pdf, true };
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
			f32 u;
			f32 v;
			if (b.x < b.y) {
				u = b.x / 2.f;
				v = b.y - u;
			} else {
				v = b.y / 2.f;
				u = b.x - v;
			}
			let w = 1.f - u - v;

			let pos = u * v0.position + v * v1.position + w * v2.position;
			let uv = u * v0.uv + v * v1.uv + w * v2.uv;
			let area = length(cross(v1.position - v0.position, v2.position - v0.position)) * 0.5f;

			let mat = instance.material;
			let em = mat->emissive.get();
			let emf = mat->emissive_factor;

			let r = pos - hit.position;
			let t = length(r);
			let dir = r / t;
			let wi = hit.to_shading(dir);
			let L = em.load(em.pixel_of_uv(uv), f32x4(1.f)).xyz * emf;
			let d = abs(dot(hit.g_normal, -wi));
			let pdf = (d != 0.f) ? (dot(dir, dir) / (area * f32(tri_count) * d)) : 0.f;

			return { L, wi, t, pdf, false };
		}
	}

	return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, false };
}

f32 light_sample_pdf() {
	return 1.f / f32(Constants.light_count + 1);
}

LightSample sample_one_light(inout Rng rng, Hit hit) {
	let n = f32(Constants.light_count + 1);

	let l = min(u32(rng.sample() * n), u32(n - 1.f));
	if (l == 0) {
		let wi = rng.sample_cos_hemi();
		let L = Constants.sky.sample(hit.position, hit.from_shading(wi));
		let pdf = (wi.z / PI) * light_sample_pdf();
		return { L, wi, 1e10f, pdf, false };
	}
	let light = Constants.lights[l - 1];
	var ls = sample_light(rng, hit, light);
	ls.pdf *= light_sample_pdf();
	return ls;
}

bool shadowed(Hit hit, LightSample ls) {
	let q = Ray(hit.ray_origin(), hit.from_shading(ls.wi), ls.t)
				.api_trace<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
	return q.CommittedStatus() != COMMITTED_NOTHING;
}

f32x3 estimate_with_light_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let ls = sample_one_light(rng, hit);
	if (any(ls.L <= 0.f) || ls.pdf <= 0.f)
		return 0.f;
	let f = eval_bsdf(hit.params, wo, ls.wi);
	if (any(f <= 0.f) || shadowed(hit, ls))
		return 0.f;

	if (ls.punctual) {
		return f * ls.L / ls.pdf;
	} else {
		let p_bounce = bsdf_pdf(hit.params, wo, ls.wi);
		let w = pow_heuristic(1, p_bounce, 1, ls.pdf);
		return w * f * ls.L / ls.pdf;
	}
}

f32x3 li(inout Rng rng, Ray ray) {
	f32x3 L = f32x3(0.f);
	f32x3 b = f32x3(1.f);
	var specular = false;
	var p_bounce = 0.f;
	var prev_hit_pos = f32x3(0.f);

	for (int bounces = 0; bounces < 10; bounces++) {
		let h = trace_ray(ray);
		if (!h.hasValue) {
			let le = Constants.sky.sample_primary(ray.origin, ray.dir);
			if (bounces == 0 || specular) {
				L += b * le;
			} else {
				// MIS for the sky light.
				let p_light = light_sample_pdf() / (4.f * PI);
				let w = pow_heuristic(1, p_bounce, 1, p_light);
				L += w * b * le;
			}
			break;
		}
		let hit = h.value;

		let le = hit.emissive;
		if (bounces == 0 || specular) {
			L += b * le;
		} else {
			// MIS for area light.
			// TODO: this should take the solid angle of the triangle into account.
			let d = hit.position - prev_hit_pos;
			let u = abs(dot(hit.g_normal, -ray.dir)) * hit.area;
			let p_light = (u != 0.f) ? (dot(d, d) / u) : 0.f;
			let w = pow_heuristic(1, p_bounce, 1, p_light);
			L += w * b * le;
		}
		// TODO: disable MIS if the ray hits a light without a bsdf?

		let wo = hit.to_shading(-ray.dir);
		if (hit.params.metallic < 1.f && hit.params.roughness > 0.f)
			L += b * estimate_with_light_sample(rng, hit, wo);

		let bs = sample_bsdf(rng, hit.params, wo);
		b *= bs.L;
		p_bounce = bs.pdf;
		ray = Ray(hit.ray_origin(), hit.from_shading(bs.wi));
		specular = bs.is_specular;
		prev_hit_pos = hit.position;

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
