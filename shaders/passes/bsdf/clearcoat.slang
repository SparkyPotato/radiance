module clearcoat;

import graph;
import graph.util.rng;
import passes.bsdf.common;
import passes.bsdf.ggx;

public struct Clearcoat {
	public f32 eval;
	public f32 r_pdf;
	public f32 f_pdf;

	__init(f32 clearcoat, f32 alpha, f32x3 wo, f32x3 wm, f32x3 wi) {}
}

f32 gtr1(f32 abs_h_l, f32 a) {
	if (a >= 1.f)
		return 1.f / PI;
	let a2 = a * a;
	return (a2 - 1.f) / (PI * log2(a2) * (1.f + (a2 - 1.f) * abs_h_l * abs_h_l));
}

public Clearcoat eval_clearcoat(f32 clearcoat, f32 alpha, f32x3 wo, f32x3 wm, f32x3 wi) {
	Clearcoat ret;

	if (clearcoat <= 0.f) {
		ret.eval = 0.f;
		ret.r_pdf = 0.f;
		ret.f_pdf = 0.f;
		return ret;
	}

	let abs_n_h = brdf_abscos(wm);
	let abs_n_l = brdf_abscos(wi);
	let abs_n_v = brdf_abscos(wo);
	let h_l = dot(wm, wi);

	let d = gtr1(abs_n_h, lerp(0.1f, 0.001, alpha));
	let f = shlick(0.04f, h_l);
	let gl = ggx_g1(wi, 0.25f);
	let gv = ggx_g1(wo, 0.25f);

	ret.r_pdf = d / (4.f * abs(dot(wi, wm)));
	ret.f_pdf = d / (4.f * abs(dot(wo, wm)));
	ret.eval = 0.25f * clearcoat * d * f * gl * gv;
}
