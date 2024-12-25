module brdf;

import graph;
import graph.util.rng;

public struct BsdfSample {
	public f32x3 L;
	public f32x3 wi;
	public bool is_specular;
}

public struct ShadingParams {
	public f32x3 base_color;
	public f32 metallic;
	public f32 roughness;
}

// clang-format off
// Resources:
// PBRT of course: https://pbr-book.org/4ed/contents
// [0] Self Shadow: https://blog.selfshadow.com/links/
// [1] Cook Torrance: https://graphics.pixar.com/library/ReflectanceModel/paper.pdf
// [2] Disney PBR: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] GGX: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// [4] Filament: https://google.github.io/filament/Filament.md.html#materialsystem
// [5] Eric Heitz' Smith extension: https://jcgt.org/published/0003/02/03/paper.pdf
// [6] Naty Hoffman Notes: https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pd
// [7] Implenting Disney BRDF: https://schuttejoe.github.io/post/disneybsdf/
// [8] Karis reference: https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// [9] https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.cpp
// clang-format on

f32 brdf_cos(f32x3 v) {
	return v.z;
}

f32 brdf_cos2(f32x3 v) {
	return v.z * v.z;
}

f32 brdf_abscos(f32x3 v) {
	return abs(v.z);
}

f32 brdf_sin2(f32x3 v) {
	return max(0.f, 1.f - brdf_cos2(v));
}

f32 brdf_sin(f32x3 v) {
	return sqrt(brdf_sin2(v));
}

f32 brdf_tan2(f32x3 v) {
	return brdf_sin2(v) / brdf_cos2(v);
}

f32 brdf_tan(f32x3 v) {
	return brdf_sin(v) / brdf_cos(v);
}

f32 brdf_cos_phi(f32x3 v) {
	let sin = brdf_sin(v);
	return (sin == 0.f) ? 0.f : v.x / sin;
}

f32 brdf_cos2_phi(f32x3 v) {
	let c = brdf_cos_phi(v);
	return c * c;
}

f32 brdf_sin_phi(f32x3 v) {
	let sin = brdf_sin(v);
	return (sin == 0.f) ? 0.f : v.y / sin;
}

f32 brdf_sin2_phi(f32x3 v) {
	let c = brdf_sin_phi(v);
	return c * c;
}

f32x3 eval_lambert(ShadingParams params, f32x3 wo, f32x3 wi) {
	let diffuse_color = (1.f - params.metallic) * params.base_color;
	return diffuse_color / PI;
}

BsdfSample sample_lambert(inout Rng rng, ShadingParams params, f32x3 wo) {
	// L = (col * n dot l / PI) / pdf
	// pdf = cos / PI
	// n dot l = cos
	// thus, L = col
	let diffuse_color = (1.f - params.metallic) * params.base_color;
	return { diffuse_color, rng.sample_cos_hemi(), false };
}

f32 v_ggx(f32x3 w, f32 roughness) {
	let a2 = roughness * roughness;
	let v = a2 * brdf_tan2(w);
	return 0.5f / max(sqrt(1.f + v) - 1.f, 1e-5f);
}

f32 g1_ggx(f32x3 wm, f32x3 wo, f32 roughness) {
	if (dot(wo, wm) * wm.z <= 0.f)
		return 0.f;
	return 1.f / (1.f + v_ggx(wo, roughness));
}

f32 g2_ggx(f32x3 wi, f32x3 wm, f32x3 wo, f32 roughness) {
	if (dot(wi, wm) * wi.z <= 0.f || dot(wo, wm) * wo.z <= 0.f)
		return 0.f;
	return 1.f / (1.f + v_ggx(wi, roughness) + v_ggx(wo, roughness));
}

f32x3 shlick(f32 u, f32x3 f0) {
	let f = pow(saturate(1.f - u), 5.f);
	return f + f0 * (1.f - f);
}

f32x3 sample_ggx_vndf(inout Rng rng, f32x3 wo, f32 roughness) {
	let u = rng.sample2();
	let vh = normalize(f32x3(roughness * wo.xy, wo.z));
	let lensq = dot(vh.xy, vh.xy);
	let T1 = lensq > 0.f ? f32x3(-vh.y, vh.x, 0.f) / sqrt(lensq) : f32x3(1.f, 0.f, 0.f);
	let T2 = cross(vh, T1);

	let r = sqrt(u.x);
	let phi = 2.f * PI * u.y;
	let t1 = r * cos(phi);
	var t2 = r * sin(phi);
	let s = 0.5f * (1.f + vh.z);
	t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

	let nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * vh;
	return normalize(f32x3(roughness * nh.xy, max(0.f, nh.z)));
}

f32 d_ggx(f32x3 wm, f32 roughness) {
	let a2 = roughness * roughness;
	let k = wm.z * wm.z * (a2 - 1.f) + 1.f;
	return a2 / (PI * k * k);
}

f32 dv_ggx(f32x3 wm, f32x3 wo, f32 roughness) {
	let l_h = dot(wm, wo);
	let g1 = g1_ggx(wm, wo, roughness);
	let d = d_ggx(wm, roughness);
	return (g1 * l_h * d) / wo.z;
}

f32x3 eval_ggx(ShadingParams params, f32x3 wo, f32x3 wi) {
	let wm = normalize(wo + wi);
	let f0 = lerp(f32x3(0.04f), params.base_color, params.metallic);
	let l_h = dot(wm, wo);

	let g1 = g1_ggx(wm, wo, params.roughness);
	let g2 = g2_ggx(wi, wm, wo, params.roughness);
	let f = shlick(l_h, f0);

	if (g1 == 0.f)
		return f32x3(0.f);
	return f * g2 / g1;
}

BsdfSample sample_ggx(inout Rng rng, ShadingParams params, f32x3 wo) {
	let wm = sample_ggx_vndf(rng, wo, params.roughness);
	let wi = reflect(-wo, wm);
	let l_h = dot(wm, wo);
	let pdf = dv_ggx(wo, wm, params.roughness) / (4.f * l_h);
	let L = eval_ggx(params, wo, wi);
	return { L, wi, true };
}

public f32x3 eval_bsdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let n_l = brdf_cos(wi);
	let fd = eval_lambert(params, wo, wi);
	let fr = eval_ggx(params, wo, wi);
	return (fd + fr) * abs(n_l);
}

public BsdfSample sample_bsdf(inout Rng rng, ShadingParams params, f32x3 wo) {
	let lambert_chance = lerp(0.5f, 0.f, params.metallic);
	BsdfSample ret;
	if (rng.sample() < lambert_chance) {
		ret = sample_lambert(rng, params, wo);
		ret.L /= lambert_chance;
	} else {
		ret = sample_ggx(rng, params, wo);
		ret.L /= 1.f - lambert_chance;
	}
	return ret;
}
