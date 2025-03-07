module brdf;

// Everything has N dot L pre-multiplied, as well as the pdf divide if required.

import graph;
import graph.util.rng;

public struct BsdfSample {
	public f32x3 f;
	public f32x3 wi;
	public f32 pdf;
	public bool is_specular;
}

public struct ShadingParams {
	public f32x3 base_color;
	public f32 metallic;
	public f32 roughness;
	public Tex2D<f32> ggx_energy_compensation_lut;
	public Sampler lut_sampler;
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

f32 pdf_lambert(f32x3 wi) {
	return brdf_cos(wi) / PI;
}

f32x3 sample_lambert(inout Rng rng, ShadingParams params, f32x3 wo) {
	return rng.sample_cos_hemi();
}

f32x3 shlick(f32 cos_t, f32x3 f0) {
	let k = 1.f - cos_t;
	let k2 = k * k;
	return f0 + (f32x3(1.f) - f0) * k2 * k2 * k;
}

f32 fresnel(f32 cos_t, f32 ior) {
	cos_t = saturate(cos_t);
	let sin2_tt = (1.f - cos_t * cos_t) / (ior * ior);
	if (sin2_tt > 1.f)
		return 1.f;

	let cos_tt = sqrt(1.f - sin2_tt);
	let par = (ior * cos_t - cos_tt) / (ior * cos_t + cos_tt);
	let perp = (cos_t - ior * cos_tt) / (cos_t + ior * cos_tt);
	return 0.5f * (par * par + perp * perp);
}

f32 d_ggx(f32x3 wm, f32 rough) {
	let n_h = brdf_cos(wm);
	let a2 = rough * rough;
	let f = (n_h * a2 - n_h) * n_h + 1.f;
	return a2 / (PI * f * f);
}

f32 v_ggx(f32x3 wo, f32x3 wi, f32 rough) {
	let n_v = brdf_cos(wo) + 1e-5f;
	let n_l = brdf_cos(wi);
	let a2 = rough * rough;
	let l = n_v * sqrt((-n_l * a2 + n_l) * n_l + a2);
	let v = n_v * sqrt((-n_v * a2 + n_v) * n_v + a2);
	return 0.5f / (l + v);
}

f32 dv_ggx(f32x3 wo, f32x3 wm, f32 rough) {
	if (brdf_cos(wm) <= 0.f)
		return 0.f;
	let g1 = g1_ggx(wo, wm, rough);
	let d = d_ggx(wm, rough);
	return g1 * max(dot(wo, wm), 0.f) * d / abs(brdf_cos(wo));
}

f32 lambda_ggx(f32x3 w, f32 rough) {
	let cos2 = brdf_cos2(w);
	let sin2 = max(0.f, 1.f - cos2);
	let tan2 = sin2 / cos2;
	let rough2 = rough * rough;
	return max(0.f, sqrt(1.f + rough2 * tan2) - 1.f) * 0.5f;
}

f32 g1_ggx(f32x3 w, f32x3 wm, f32 rough) {
	if (dot(w, wm) * brdf_cos(w) <= 0.f)
		return 0.f;
	return 1.f / (1.f + lambda_ggx(w, rough));
}

f32 g2_ggx(f32x3 wo, f32x3 wm, f32x3 wi, f32 rough) {
	if (dot(wo, wm) * brdf_cos(wo) <= 0.f || dot(wi, wm) * brdf_cos(wi) <= 0.f)
		return 0.f;
	return 1.f / (1.f + lambda_ggx(wi, rough) + lambda_ggx(wo, rough));
}

// Sampling with spherical caps:
// https://arxiv.org/abs/2306.05044
f32x3 sample_dv_ggx(inout Rng rng, f32x3 wo, f32 rough) {
	let u = rng.sample2();
	let vh = normalize(f32x3(rough * wo.xy, wo.z));

	let phi = 2.f * PI * u.x;
	let z = fma(1.f - u.y, 1.f + vh.z, -vh.z);
	let sin_theta = sqrt(clamp(1.f - z * z, 0.f, 1.f));
	let x = sin_theta * cos(phi);
	let y = sin_theta * sin(phi);
	let nh = f32x3(x, y, z) + vh;

	return normalize(f32x3(rough * nh.xy, max(0.f, nh.z)));
}

f32x3 eval_ggx(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	if (params.roughness < 0.001f)
		return f32x3(0.f);

	let f0 = lerp(f32x3(0.04f), params.base_color, params.metallic);
	let f = shlick(abs(dot(wo, wm)), f0);
	let d = d_ggx(wm, params.roughness);
	let v = v_ggx(wo, wi, params.roughness);
	let rho_ss = d * v * f;

	let n_v = brdf_cos(wo);
	let e_ss = params.ggx_energy_compensation_lut.sample_mip(params.lut_sampler, f32x2(n_v, params.roughness), 0.f);
	let m_ss = f0 * (1.f - e_ss) / e_ss;

	return rho_ss * (1.f + m_ss);
}

f32 pdf_ggx(f32x3 wm, f32x3 wo, f32 rough) {
	if (rough < 0.001f)
		return 0.f;

	return dv_ggx(wo, wm, rough) / (4.f * abs(dot(wm, wo)));
}

f32x3 sample_ggx(inout Rng rng, ShadingParams params, f32x3 wo) {
	let wm = sample_dv_ggx(rng, wo, params.roughness);
	return reflect(-wo, wm);
}

public f32x3 eval_bsdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let fd = eval_lambert(params, wo, wi);
	let wm = normalize(wo + wi);
	let fr = eval_ggx(params, wo, wm, wi);
	return (fd + fr) * abs(brdf_cos(wi));
}

public BsdfSample sample_bsdf(inout Rng rng, ShadingParams params, f32x3 wo) {
	let lambert_chance = lerp(lerp(0.4f, 0.9f, params.roughness), 0.f, params.metallic);
	let spec_chance = 1.f - lambert_chance;
	let perfectly_specular = params.roughness < 0.001f;
	let lambert_sel = rng.sample() < lambert_chance;

	let wi = lambert_sel ? sample_lambert(rng, params, wo)
						 : (perfectly_specular ? f32x3(-wo.xy, wo.z) : sample_ggx(rng, params, wo));
	let wm = normalize(wo + wi);

	let fd = eval_lambert(params, wo, wi);
	let pdf_d = pdf_lambert(wi);
	let fr = perfectly_specular ? lerp(f32x3(0.04f), params.base_color, params.metallic) : eval_ggx(params, wo, wm, wi);
	let pdf_r = perfectly_specular ? 1.f : pdf_ggx(wm, wo, params.roughness);

	let pdf = pdf_d * lambert_chance + pdf_r * spec_chance;
	let w = lambert_sel ? bal_heuristic_1(pdf_d, pdf_r) : bal_heuristic_1(pdf_r, pdf_d);
	let f = w * (fd + fr) * abs(brdf_cos(wi)) / pdf;

	return { f, wi, pdf, perfectly_specular && lambert_chance == 0.f };
}

public f32 bsdf_pdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let lambert_chance = lerp(0.5f, 0.f, params.metallic);
	let spec_chance = 1.f - lambert_chance;
	let lambert_pdf = lambert_chance * pdf_lambert(wi);
	let wm = normalize(wi + wo);
	let ggx_pdf = spec_chance * pdf_ggx(wm, wo, params.roughness);
	return lambert_pdf + ggx_pdf;
}
