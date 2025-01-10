import graph;
import graph.util.rng;
import asset;
import passes.bsdf;
import common;

struct WorldVertex {
	f32x3 position;
	f32x2 uv;
}

struct WorldTriHit {
	WorldVertex v0;
	WorldVertex v1;
	WorldVertex v2;
	f32x3 position;
	f32x2 uv;
	f32x3 normal;
	f32 area;

	__init(u32 instance, u32 tri, f32x2 b) {
		let instance = Constants.instances[instance];
		let iptr = (u32*)(instance.raw_mesh + instance.raw_vertex_count);
		let i0 = tri * 3;
		let v0 = instance.raw_mesh[iptr[i0 + 0]];
		let v1 = instance.raw_mesh[iptr[i0 + 1]];
		let v2 = instance.raw_mesh[iptr[i0 + 2]];
		let bary = f32x3(1.f - (b.x + b.y), b.x, b.y);

		let tmat = instance.transform.mat();
		this.v0 = { mul(tmat, f32x4(v0.position, 1.f)).xyz, v0.uv };
		this.v1 = { mul(tmat, f32x4(v1.position, 1.f)).xyz, v1.uv };
		this.v2 = { mul(tmat, f32x4(v2.position, 1.f)).xyz, v2.uv };

		let v0p = this.v0.position;
		let v1p = this.v1.position;
		let v2p = this.v2.position;

		this.position = bary.x * v0p + bary.y * v1p + bary.z * v2p;
		this.uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
		let obj_norm = f32x4(bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal, 0.f);
		this.normal = normalize(mul(tmat, obj_norm).xyz);
		this.area = length(cross(v1p - v0p, v2p - v0p)) * 0.5f;
	}
}

struct Tbn {
	f32x3 tangent;
	f32x3 bitangent;
	f32x3 normal;

	__init(WorldVertex v0, WorldVertex v1, WorldVertex v2, f32x3 n) {
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

struct Hit {
	f32x3 position;
	f32x3 g_normal;
	f32x3 emissive;
	f32x3x3 to_shading_basis;
	f32x3x3 from_shading_basis;
	ShadingParams params;
	f32 area;

	__init(BuiltInTriangleIntersectionAttributes attrs) {
		let thit = WorldTriHit(InstanceIndex(), PrimitiveIndex(), attrs.barycentrics);
		this.position = thit.position;
		this.g_normal = thit.normal;
		this.area = thit.area;

		let tbn = Tbn(thit.v0, thit.v1, thit.v2, this.g_normal);
		this.to_shading_basis = f32x3x3(tbn.tangent, tbn.bitangent, tbn.normal);
		this.from_shading_basis = transpose(this.to_shading_basis);

		let mat = Constants.instances[InstanceIndex()].material;
		let s = Constants.sampler;
		let bc = mat->base_color.get();
		let mr = mat->metallic_roughness.get();
		let nm = mat->normal.get();
		let em = mat->emissive.get();
		let white = f32x4(1.f);
		let blue = f32x4(0.f, 0.f, 1.f, 1.f);

		this.params.base_color = (bc.sample(s, thit.uv, white) * mat->base_color_factor).xyz;
		let met_rough = mr.sample(s, thit.uv, white);
		let rough = met_rough.y * mat->roughness_factor;
		this.params.metallic = met_rough.z * mat->metallic_factor;
		this.params.roughness = max(rough * rough, 0.0001f);
		this.emissive = em.sample(s, thit.uv, white).xyz * mat->emissive_factor;

		let norm_world = this.from_shading(nm.sample(s, thit.uv, blue).xyz);
		this.from_shading_basis._m02_m12_m22 = norm_world;
		this.to_shading_basis._m20_m21_m22 = norm_world;
	}

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
			// TODO: sample disk and figure out atmosphere transmittance correctly.
			let dir = -light.pos_or_dir;
			let L = light.radiance * Constants.sky.sun_transmittance(hit.position, dir);
			let wi = hit.to_shading(dir);
			let pdf = solid_angle_pdf(radians(0.5f));
			return { L, wi, 1e10f, pdf, true };
		}
		case LightType.Emissive: {
			let i = asuint(light.radiance.x);
			let instance = Constants.instances[i];
			let tri_count = instance.raw_tri_count;
			let tri = min(u32(rng.sample() * tri_count), tri_count - 1);

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
			let thit = WorldTriHit(i, tri, f32x2(u, v));

			let mat = instance.material;
			let em = mat->emissive.get();
			let emf = mat->emissive_factor;
			let L = em.sample(Constants.sampler, thit.uv, f32x4(1.f)).xyz * emf;

			let r = thit.position - hit.position;
			let t = length(r);
			let dir = r / t;
			let wi = hit.to_shading(dir);
			let d = abs(dot(hit.g_normal, -wi));
			let pdf = (d != 0.f) ? (dot(dir, dir) / (thit.area * f32(tri_count) * d)) : 0.f;

			return { L, wi, t, pdf, false };
		}
	}

	return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, false };
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

Ray shadow_ray(Hit hit, LightSample ls) {
	return Ray(hit.ray_origin(), hit.from_shading(ls.wi), ls.t - 2e-5f);
}

struct SampleEstimate {
	f32x3 L;
	Ray ray;
}

SampleEstimate estimate_with_light_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let ls = sample_one_light(rng, hit);
	if (any(ls.L <= 0.f) || ls.pdf <= 0.f)
		return { f32x3(0.f), Ray() };
	let f = eval_bsdf(hit.params, wo, ls.wi);
	if (any(f <= 0.f))
		return { f32x3(0.f), Ray() };

	let ray = shadow_ray(hit, ls);
	var L = f * ls.L / ls.pdf;
	if (!ls.punctual) {
		let p_bounce = bsdf_pdf(hit.params, wo, ls.wi);
		L *= pow_heuristic(1, p_bounce, 1, ls.pdf);
	}
	return { L, ray };
}

[shader("closesthit")]
void main(inout HitPayload p, BuiltInTriangleIntersectionAttributes attrs) {
	p.hit = true;
	let hit = Hit(attrs);

	let le = hit.emissive;
	if (p.specular) {
		p.L += p.b * le;
	} else {
		// MIS for area light.
		// TODO: this should take the solid angle of the triangle into account.
		let d = hit.position - p.prev_hit_pos;
		let u = abs(dot(hit.g_normal, -p.ray.dir)) * hit.area;
		let p_light = (u != 0.f) ? (dot(d, d) / u) : 0.f;
		let w = pow_heuristic(1, p.p_bounce, 1, p_light);
		p.L += p.b * w * le;
	}
	// TODO: disable MIS if the ray hits a light without a bsdf?

	let wo = hit.to_shading(-p.ray.dir);
	if (hit.params.metallic < 1.f || hit.params.roughness > 0.f) {
		let es = estimate_with_light_sample(p.rng, hit, wo);
		p.L_light_sample = p.b * es.L;
		p.shadow_ray = es.ray;
	}

	let bs = sample_bsdf(p.rng, hit.params, wo);
	p.b *= bs.L;
	p.p_bounce = bs.pdf;
	p.ray = Ray(hit.ray_origin(), hit.from_shading(bs.wi));
	p.specular = bs.is_specular;
	p.prev_hit_pos = hit.position;
}
