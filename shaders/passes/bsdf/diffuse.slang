module diffuse;

import graph;
import graph.util.rng;
import passes.bsdf.common;
import passes.bsdf.sheen;

f32 eval_retro_diffuse(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	let n_l = brdf_abscos(wi);
	let n_v = brdf_abscos(wo);

	let roughness = params.roughness * params.roughness;
	let rr = 0.5f + 2.f * n_l * n_l * roughness;
	let fl = shlick_weight(n_l);
	let fv = shlick_weight(n_v);

	return rr * (fl + fv + fl * fv * (rr - 1.f));
}

public f32 eval_diffuse(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi, bool thin) {
	let n_l = brdf_abscos(wi);
	let n_v = brdf_abscos(wo);

	let fl = shlick_weight(n_l);
	let fv = shlick_weight(n_v);

	var hanrahan_krueger = 0.f;
	if (thin && params.flatness > 0.f) {
		let roughness = params.roughness * params.roughness;
		let h_l = dot(wm, wi);
		let fss90 = h_l * h_l * roughness;
		let fss = lerp(1.f, fss90, fl) * lerp(1.f, fss90, fv);
		hanrahan_krueger = 1.25f * (fss * (1.f / (n_l + n_v) - 0.5f) + 0.5f);
	}

	let lambert = 1.f;
	let retro = eval_retro_diffuse(params, wo, wm, wi);
	let subsurf = lerp(lambert, hanrahan_krueger, thin ? params.flatness : 0.f);

	return (retro + subsurf * (1.f - 0.5f * fl) * (1.f - 0.5f * fv)) / PI;
}

public BsdfSample sample_diffuse(inout Rng rng, ShadingParams params, f32x3 wo, bool thin) {
	let sign = sign(brdf_cos(wo));
	let wi = sign * rng.sample_cos_hemi();
	let wm = normalize(wi + wo);

	let n_l = brdf_cos(wi);
	if (n_l == 0.f) {
		return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, false };
	}
	let n_v = brdf_cos(wo);

	// TODO: diffuse transmittance
	let color = params.base_color;
	let sheen = eval_sheen(params, wo, wm, wi);
	let diffuse = eval_diffuse(params, wo, wm, wi, thin);
	return { sheen + color * diffuse, wi, abs(n_v), abs(n_l), false };
}
