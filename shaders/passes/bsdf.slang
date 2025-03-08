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
// [0] https://blog.selfshadow.com/links/
// [1] Cook Torrance: https://graphics.pixar.com/library/ReflectanceModel/paper.pdf
// [2] Disney PBR: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.cpp
// [3] https://schuttejoe.github.io/post/disneybsdf/
// [5] GGX: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// [6] Filament: https://google.github.io/filament/Filament.md.html#materialsystem
// [7] https://jcgt.org/published/0003/02/03/paper.pdf
// [8] https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pd
// [9] https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// [10] Bounded VNDF sampling: https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
// [11] Multiscatter-GGX: https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
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

f32 d_ggx(f32x3 wm, f32 rough) {
	let tan2 = brdf_tan2(wm);
	if (isinf(tan2))
		return 0.f;
	let cos2 = brdf_cos2(wm);
	let cos4 = cos2 * cos2;
	let cos_phi = brdf_cos_phi(wm) / rough;
	let sin_phi = brdf_sin_phi(wm) / rough;
	let e = tan2 * (cos_phi * cos_phi + sin_phi * sin_phi);
	return 1.f / (PI * rough * rough * cos4 * (1.f + e) * (1.f + e));

	// let n_h2 = brdf_cos2(wm);
	// let denom = n_h2 * (a2 - 1.f) + 1.f;
	// return a2 / (PI * denom * denom);
}

f32 lambda_ggx(f32x3 w, f32 rough) {
	let tan2 = brdf_tan2(w);
	if (isinf(tan2))
		return 0.f;
	let cos_phi = brdf_cos_phi(w) * rough;
	let sin_phi = brdf_sin_phi(w) * rough;
	let a2 = cos_phi * cos_phi + sin_phi * sin_phi;
	return (sqrt(1.f + a2 * tan2) - 1.f) / 2.f;
}

f32 g_ggx(f32x3 wo, f32x3 wi, f32 rough) {
	return 1.f / (1.f + lambda_ggx(wo, rough) + lambda_ggx(wi, rough));
}

f32 g1_ggx(f32x3 w, f32 rough) {
	let a2 = rough * rough;
	let n_v = abs(brdf_cos(w));
	return 2.f / (1.f + sqrt(a2 + (1.f - a2) * n_v * n_v));
}

f32x3 eval_ggx(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	if (params.roughness < 0.001f)
		return f32x3(0.f);

	let n_l = abs(brdf_cos(wi));
	let n_v = abs(brdf_cos(wo));
	let f0 = lerp(f32x3(0.04f), params.base_color, params.metallic);
	let f = shlick(abs(dot(wo, wm)), f0);
	let d = d_ggx(wm, params.roughness);
	let g = g_ggx(wo, wi, params.roughness);
	let rho_ss = d * f * g / (4.f * n_l * n_v);
	// let gl = g1_ggx(wi, params.roughness);
	// let gv = g1_ggx(wo, params.roughness);
	// let rho_ss = d * gl * gv * f / (4.f * n_l * n_v);

	let e_ss = params.ggx_energy_compensation_lut.sample_mip(params.lut_sampler, f32x2(n_v, params.roughness), 0.f);
	let m_ss = f0 * (1.f - e_ss) / e_ss;

	return rho_ss * (1.f + m_ss);
}

f32 pdf_ggx(f32x3 wo, f32x3 wm, f32x3 wi, f32 rough) {
	if (rough < 0.001f)
		return 0.f;

	let d = d_ggx(wm, rough);
	let ai = wo.xy * rough;
	let len2 = dot(ai, ai);
	let t = sqrt(len2 + wo.z * wo.z);
	if (wi.z >= 0.f) {
		let s = 1.f + length(wo.xy);
		let a2 = rough * rough;
		let s2 = s * s;
		let k = (1.f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
		return d / (2.f * (k * wo.z + t));
	}
	return d * (t - wo.z) / (2.f * len2);
}

f32x3 sample_ggx(inout Rng rng, ShadingParams params, f32x3 wo) {
	let rough = params.roughness;
	let u = rng.sample2();
	let vh = normalize(f32x3(rough * wo.xy, wo.z));

	let phi = 2.f * PI * u.x;
	let s = 1.f + length((wo.xy));
	let a2 = rough * rough;
	let s2 = s * s;
	let k = (1.f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
	let b = wo.z > 0.f ? k * vh.z : vh.z;
	let z = fma(1.f - u.y, 1.f + b, -b);
	let sin_theta = sqrt(saturate(1.f - z * z));
	let x = sin_theta * cos(phi);
	let y = sin_theta * sin(phi);
	let nh = f32x3(x, y, z) + vh;
	let n = normalize(f32x3(rough * nh.xy, max(0.f, nh.z)));
	return 2.f * dot(wo, n) * n - wo;
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
	if (wi.z < 0.f)
		return { f32x3(0.f), wi, 1.f, false };
	let wm = normalize(wo + wi);

	let fd = eval_lambert(params, wo, wi);
	let pdf_d = pdf_lambert(wi);
	let pdf_r = perfectly_specular ? 1.f : pdf_ggx(wo, wm, wi, params.roughness);
	let fr = clamp(
		perfectly_specular ? lerp(f32x3(0.04f), params.base_color, params.metallic) : eval_ggx(params, wo, wm, wi), 0.f,
		10.f);

	let pdf = pdf_d * lambert_chance + pdf_r * spec_chance;
	let w = lambert_sel ? bal_heuristic_1(pdf_d, pdf_r) : bal_heuristic_1(pdf_r, pdf_d);
	let f = w * (fd + fr) * abs(brdf_cos(wi)) / pdf;

	return { f, wi, pdf, perfectly_specular && lambert_chance == 0.f };
}

public f32 bsdf_pdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let lambert_chance = lerp(lerp(0.4f, 0.9f, params.roughness), 0.f, params.metallic);
	let spec_chance = 1.f - lambert_chance;
	let perfectly_specular = params.roughness < 0.001f;

	let pdf_d = lambert_chance * pdf_lambert(wi);
	let wm = normalize(wi + wo);
	let pdf_r = spec_chance * (perfectly_specular ? 1.f : pdf_ggx(wo, wm, wi, params.roughness));
	return pdf_d + pdf_r;
}
