module brdf;

import graph;
import graph.util.rng;

public struct BsdfSample {
	// note: is premultiplied by cos(theta) / pdf.
	public f32x3 L;
	public f32x3 wi;
	public f32 pdf;
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
	let wi = rng.sample_cos_hemi();
	return { diffuse_color, wi, lambert_pdf(wi), false };
}

f32 lambert_pdf(f32x3 wi) {
	return brdf_cos(wi) / PI;
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

// Sampling with spherical caps:
// https://arxiv.org/abs/2306.05044
f32x3 sample_ggx_vndf(inout Rng rng, f32x3 wo, f32 roughness) {
	let u = rng.sample2();
	let vh = normalize(f32x3(roughness * wo.xy, wo.z));

	let phi = 2.f * PI * u.x;
	let z = fma(1.f - u.y, 1.f + vh.z, -vh.z);
	let sin_theta = sqrt(clamp(1.f - z * z, 0.f, 1.f));
	let x = sin_theta * cos(phi);
	let y = sin_theta * sin(phi);
	let nh = f32x3(x, y, z) + vh;

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
	let L = eval_ggx(params, wo, wi);
	let pdf = ggx_pdf(wo, wm, params.roughness);
	return { L, wi, pdf, true };
}

f32 ggx_pdf(f32x3 wo, f32x3 wm, f32 roughness) {
	let l_h = dot(wm, wo);
	return dv_ggx(wo, wm, roughness) / (4.f * l_h);
}

public f32 pow_heuristic(u32 nf, f32 f_pdf, u32 ng, f32 g_pdf) {
	let f = f_pdf * f32(nf);
	let g = g_pdf * f32(ng);
	return f * f / (f * f + g * g);
}

public f32x3 eval_bsdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let n_l = brdf_cos(wi);
	let fd = eval_lambert(params, wo, wi);
	let fr = eval_ggx(params, wo, wi);
	return (fd + fr) * abs(n_l);
}

public BsdfSample sample_bsdf(inout Rng rng, ShadingParams params, f32x3 wo) {
	let lambert_chance = lerp(0.5f, 0.f, params.metallic);
	let spec_chance = 1.f - lambert_chance;
	BsdfSample ret;
	if (rng.sample() < lambert_chance) {
		ret = sample_lambert(rng, params, wo);
		ret.L /= lambert_chance;
		ret.pdf *= lambert_chance;
	} else {
		ret = sample_ggx(rng, params, wo);
		ret.L /= spec_chance;
		ret.pdf *= spec_chance;
	}
	return ret;
}

public f32 bsdf_pdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let lambert_chance = lerp(0.5f, 0.f, params.metallic);
	let lambert_pdf = lambert_chance * lambert_pdf(wi);
	let wm = normalize(wi + wo);
	let ggx_pdf = (1.f - lambert_chance) * ggx_pdf(wo, wm, params.roughness);
	return lambert_pdf + ggx_pdf;
}
