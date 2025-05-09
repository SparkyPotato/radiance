import graph;
import graph.util.color;
import graph.util.rng;
import asset;
import passes.bsdf;
import common;

struct WorldVertex {
	f32x3 position;
	f32x2 uv;
}

f32x3[3] get_positions() {
	return spirv_asm {
		result:$$f32x3[3] = OpLoad builtin(HitTriangleVertexPositionsKHR:f32x3[3])
	};
}

struct WorldTriHit {
	WorldVertex v0;
	WorldVertex v1;
	WorldVertex v2;
	f32x3 position;
	f32x2 uv;
	f32x3 normal;
	f32x3 g_normal;
	f32 area;
	u32 tri_count;

	[ForceInline]
	__init(u32 instance, u32 tri, f32x2 b) {
		let instance = &Constants.instances[instance];
		let i0 = tri * 3;
		let bary = f32x3(1.f - (b.x + b.y), b.x, b.y);

		let iptr = (u32*)(instance->raw_mesh + instance->raw_vertex_count);
		let v0 = instance->raw_mesh[iptr[i0 + 0]];
		let v1 = instance->raw_mesh[iptr[i0 + 1]];
		let v2 = instance->raw_mesh[iptr[i0 + 2]];

		let tmat = instance->transform.mat();
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
		this.g_normal = cross(v1p - v0p, v2p - v0p);
		this.area = length(this.g_normal) * 0.5f;
		this.g_normal = normalize(this.g_normal);
		this.tri_count = instance->raw_tri_count;
	}
}

struct Tbn {
	f32x3 tangent;
	f32x3 bitangent;
	f32x3 normal;

	[ForceInline]
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

			this.tangent = t;
			this.bitangent = b;
			this.normal = n;
		} else {
			f32x3 other;
			if (n.x < 0.1f && n.y < 0.1f) {
				other = f32x3(0.f, -n.z, n.y);
			} else {
				other = f32x3(-n.y, n.x, 0.f);
			}
			other = normalize(other);
			let last = cross(other, n);

			this.tangent = other;
			this.bitangent = last;
			this.normal = n;
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
	u32 tri_count;

	__init(BuiltInTriangleIntersectionAttributes attrs) {
		let thit = WorldTriHit(InstanceIndex(), PrimitiveIndex(), attrs.barycentrics);
		this.position = thit.position;
		this.g_normal = thit.g_normal;
		this.area = thit.area;
		this.tri_count = thit.tri_count;

		let tbn = Tbn(thit.v0, thit.v1, thit.v2, thit.normal);
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

		this.params.base_color =
			rec709_to_rec2020((bc.sample_mip(s, thit.uv, 0.0, white) * mat->base_color_factor).xyz);
		let met_rough = mr.sample_mip(s, thit.uv, 0.0, white);
		let rough = met_rough.y * mat->roughness_factor;
		this.params.metallic = met_rough.z * mat->metallic_factor;
		this.params.roughness = rough * rough;
		this.emissive = rec709_to_rec2020(em.sample_mip(s, thit.uv, 0.0, white).xyz * mat->emissive_factor);

		this.params.ggx_energy_compensation_lut = Constants.ggx_energy_compensation_lut;
		this.params.lut_sampler = Constants.sampler;

		// let norm_world = this.from_shading(nm.sample_mip(s, thit.uv, 0.0, blue).xyz);
		// this.from_shading_basis._m02_m12_m22 = norm_world;
		// this.to_shading_basis._m20_m21_m22 = norm_world;
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
	f32x3 wi_world;
	f32 t;
	f32 pdf;
	bool punctual;
}

f32 solid_angle_pdf(f32 theta) {
	return 1.f / (2.f * PI * (1.f - cos(theta)));
}

[ForceInline]
LightSample sample_emissive(inout Rng rng, Hit hit, u32 i) {
	let instance = &Constants.instances[i];
	let tri_count = instance->raw_tri_count;
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

	let mat = instance->material;
	let em = mat->emissive.get();
	let emf = mat->emissive_factor;
	let L = rec709_to_rec2020(em.sample_mip(Constants.sampler, thit.uv, 0.0, f32x4(1.f)).xyz * emf);

	let r = thit.position - hit.position;
	let t2 = dot(r, r);
	let t = length(r);
	let wi = r / t;
	let d = abs(dot(thit.g_normal, -wi));
	let pdf = (d != 0.f) ? (dot(wi, wi) / (thit.area * f32(tri_count) * d)) : 0.f;

	return { L / t2, wi, t, pdf, false };
}

// TODO: shrample light (light tree) and triangles (by solid angle or area) better.
LightSample sample_light(inout Rng rng, Hit hit, Light light) {
	switch (light.ty) {
		case LightType.Point: {
			let pos = light.pos_or_dir;
			let dir = pos - hit.position;
			let t2 = dot(dir, dir);
			let L = rec709_to_rec2020(light.radiance / t2);
			let t = sqrt(t2);
			let wi = dir / t;
			return { L, wi, t, 1.f, true };
		}
		case LightType.Directional: {
			// TODO: sample disk and figure out atmosphere transmittance correctly.
			let dir = -light.pos_or_dir;
			let L = rec709_to_rec2020(light.radiance * Constants.sky.sun_transmittance(hit.position, dir));
			let disk_pdf = solid_angle_pdf(radians(0.5f));
			return { L / disk_pdf, dir, 1e10f, 1.f, true };
		}
		case LightType.Emissive: {
			let i = asuint(light.radiance.x);
			return sample_emissive(rng, hit, i);
		}
	}

	return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, false };
}

LightSample sample_sky(inout Rng rng, Hit hit) {
	let wi = rng.sample_cos_hemi();
	let dir = hit.from_shading(wi);
	let L = rec709_to_rec2020(Constants.sky.sample(hit.position, dir));
	return { L, dir, 1e10f, wi.z / PI, false };
}

LightSample sample_one_light(inout Rng rng, Hit hit) {
	let n = f32(Constants.light_count + 1);
	let l = min(u32(rng.sample() * n), u32(n - 1.f));

	LightSample ls;
	if (l == Constants.light_count) {
		ls = sample_sky(rng, hit);
	} else {
		ls = sample_light(rng, hit, Constants.lights[l]);
	}

	ls.pdf *= light_sample_pdf();
	return ls;
}

Ray shadow_ray(Hit hit, LightSample ls) {
	return Ray(hit.ray_origin(), ls.wi_world, ls.t - 2e-5f);
}

struct LightEstimate {
	f32x3 L;
	Ray shadow;
}

LightEstimate estimate_with_light_sample(inout Rng rng, Hit hit, f32x3 wo) {
	let ls = sample_one_light(rng, hit);
	let ray = shadow_ray(hit, ls);
	if (all(ls.L <= 0.f) || ls.pdf <= 0.f)
		return { f32x3(0.f), ray };
	let wi = hit.to_shading(ls.wi_world);
	let f = eval_bsdf(hit.params, wo, wi);
	if (all(f <= 0.f))
		return { f32x3(0.f), ray };
	var L = f * ls.L / ls.pdf;
	if (!ls.punctual) {
		let p_bounce = bsdf_pdf(hit.params, wo, wi);
		L *= pow_heuristic_1(ls.pdf, p_bounce);	 // Should this be the other way around?
	}
	return { L, ray };
}

[shader("closesthit")]
void main(inout HitPayload p, BuiltInTriangleIntersectionAttributes attrs) {
	let hit = Hit(attrs);
	p.prev_hit_norm = hit.from_shading(f32x3(0.f, 0.f, 1.f));

	let le = hit.emissive;
	f32 w = 1.f;
	if (!p.specular) {
		// MIS for area light.
		let p_light = light_sample_pdf() / (hit.area * f32(hit.tri_count));
		w = pow_heuristic_1(p.p_bounce, p_light);
	}
	p.L += p.b * w * le;
	// TODO: disable MIS if the ray hits a light without a bsdf?

	let wo = hit.to_shading(-p.ray.dir);

	let bs = sample_bsdf(p.rng, hit.params, wo);
	let throughput = bs.f;
	p.p_bounce = bs.pdf;
	p.ray = Ray(hit.ray_origin(), hit.from_shading(bs.wi));
	p.specular = bs.is_specular;

	// Do this as late as possible to minimize the live state kept between the shadow ray trace.
	var el = estimate_with_light_sample(p.rng, hit, wo);
	if (all(el.L <= 0.f))
		el.shadow.t = 0.f;

	// We use the same payload even though we only need a single bool
	// because otherwise the payload would have to be stored as live state.
	p.hit = true;
	el.shadow.trace(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
					1, p);
	if (!p.hit) {
		p.L += p.b * el.L;
		p.hit = true;
	}

	p.b *= throughput;
}
