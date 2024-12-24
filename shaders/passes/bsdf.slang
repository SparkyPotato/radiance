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

public f32 brdf_cos(f32x3 v) {
	return v.z;
}

public f32 brdf_cos2(f32x3 v) {
	return v.z * v.z;
}

public f32 brdf_abscos(f32x3 v) {
	return abs(v.z);
}

public f32 brdf_sin2(f32x3 v) {
	return max(0.f, 1.f - brdf_cos2(v));
}

public f32 brdf_sin(f32x3 v) {
	return sqrt(brdf_sin2(v));
}

public f32 brdf_tan2(f32x3 v) {
	return brdf_sin2(v) / brdf_cos2(v);
}

public f32 brdf_tan(f32x3 v) {
	return brdf_sin(v) / brdf_cos(v);
}

public f32 brdf_cos_phi(f32x3 v) {
	let sin = brdf_sin(v);
	return (sin == 0.f) ? 0.f : v.x / sin;
}

public f32 brdf_cos2_phi(f32x3 v) {
	let c = brdf_cos_phi(v);
	return c * c;
}

public f32 brdf_sin_phi(f32x3 v) {
	let sin = brdf_sin(v);
	return (sin == 0.f) ? 0.f : v.y / sin;
}

public f32 brdf_sin2_phi(f32x3 v) {
	let c = brdf_sin_phi(v);
	return c * c;
}

public f32x3 shlick(f32x3 r0, f32 rad) {
	let exp = pow(1.f - rad, 5.f);
	return r0 + (f32x3(1.f) - r0) * exp;
}

public f32 shlick(f32 r0, f32 rad) {
	return lerp(1.f, shlick_weight(rad), r0);
}

public f32 shlick_weight(f32 u) {
	let m = saturate(1.f - u);
	let m2 = m * m;
	return m * m2 * m2;
}

public f32 shlick_r0_from_rel_ior(f32 eta) {
	let em = eta - 1.f;
	let ep = eta + 1.f;
	return (em * em) / (ep * ep);
}

public f32x3 eval_bsdf(ShadingParams params, f32x3 wo, f32x3 wi) {
	let n_l = brdf_cos(wi);
	return params.base_color / PI * abs(n_l);
}

public BsdfSample sample_bsdf(inout Rng rng, ShadingParams params, f32x3 wo) {
	let wi = rng.sample_cos_hemi();
	let l = params.base_color / PI;
	return { l, wi, false };
}
