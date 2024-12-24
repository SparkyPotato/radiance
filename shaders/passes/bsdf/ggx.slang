module ggx;

import graph;
import graph.util.rng;
import passes.bsdf.common;
import passes.bsdf.anisotropic;
import passes.bsdf.sheen;

f32 dielectric_fresnel(f32 cos_i, f32 ni, f32 nt) {
	cos_i = clamp(cos_i, -1.f, 1.f);
	if (cos_i < 0.f) {
		let t = ni;
		ni = nt;
		nt = t;
		cos_i = -cos_i;
	}

	let sin_i = sqrt(max(0.f, 1.f - cos_i * cos_i));
	let sin_t = ni / nt * sin_i;

	if (sin_t >= 1.f)
		return 1.f;

	let cos_t = sqrt(max(0.f, 1.f - sin_t * sin_t));

	let r_par = ((nt * cos_i) - (ni * cos_t)) / ((nt * cos_i) + (ni * cos_t));
	let r_per = ((ni * cos_i) - (nt * cos_t)) / ((ni * cos_i) + (nt * cos_t));
	return (r_par * r_par + r_per * r_per) / 2.f;
}

f32 ggx_d(f32x3 wm, f32 ax, f32 ay) {
	let h_x2 = wm.x * wm.x;
	let h_y2 = wm.y * wm.y;
	let cos2 = brdf_cos2(wm);
	let ax2 = ax * ax;
	let ay2 = ay * ay;

	let s = h_x2 / ax2 + h_y2 / ay2 + cos2;
	return 1.f / (PI * ax * ay * s * s);
}

public f32 ggx_g1(f32x3 w, f32x3 wm, f32 ax, f32 ay) {
	let abs_tan = abs(brdf_tan(w));
	if (isinf(abs_tan))
		return 0.f;

	let a = sqrt(brdf_cos2_phi(w) * ax * ax + brdf_sin2_phi(w) * ay * ay);
	let a_tan = a * abs_tan;
	let a2_tan = a_tan * a_tan;

	let lambda = 0.5f * (-1.f + sqrt(1.0f + a2_tan));
	return 1.f / (1.f + lambda);
}

f32 ggx_g1(f32x3 w, f32 a) {
	let a2 = a * a;
	let abs_n_v = brdf_abscos(w);
	return 2.f / (1.f + sqrt(a2 + (1.f - a2) * abs_n_v * abs_n_v));
}

public struct GgxVAnisoPdf {
	public f32 r_pdf;
	public f32 f_pdf;
}

public GgxVAnisoPdf eval_ggx_v_aniso(f32x3 wi, f32x3 wm, f32x3 wo, f32 ax, f32 ay) {
	GgxVAnisoPdf ret;
	let d = ggx_d(wm, ax, ay);

	let abs_n_l = brdf_abscos(wi);
	let abs_h_l = abs(dot(wm, wi));
	let g1v = ggx_g1(wo, wm, ax, ay);
	ret.f_pdf = g1v * abs_h_l * d / abs_n_l;

	let abs_n_v = brdf_abscos(wo);
	let abs_h_v = abs(dot(wm, wo));
	let g1l = ggx_g1(wi, wm, ax, ay);
	ret.r_pdf = g1l * abs_h_v * d / abs_n_v;

	return ret;
}

public f32x3 sample_ggx_v_aniso(f32x3 wo, f32 ax, f32 ay, f32x2 u) {
	let v = normalize(f32x3(wo.x * ax, wo.y * ay, wo.z));

	let t1 = (v.z < 0.9999f) ? normalize(cross(v, f32x3(0.f, 0.f, 1.f))) : f32x3(1.f, 0.f, 0.f);
	let t2 = cross(t1, v);

	let a = 1.f / (1.f + v.z);
	let r = sqrt(u.x);
	let phi = (u.y < a) ? (u.y / a) * PI : PI + (u.y - a) / (1.f - a) * PI;
	let p1 = r * cos(phi);
	let p2 = r * sin(phi) * ((u.y < a) ? 1.f : v.z);

	let n = p1 * t1 + p2 * t2 + sqrt(max(0.f, 1.f - p1 * p1 - p2 * p2)) * v;
	return normalize(f32x3(ax * n.x, ay * n.y, n.z));
}

f32x3 fresnel(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	let h_v = dot(wm, wo);

	let tint = tint(params.base_color);

	var r0 = shlick_r0_from_rel_ior(params.rel_ior) * lerp(f32x3(1.f), tint, params.spec_tint);
	r0 = lerp(r0, params.base_color, params.metallic);

	let dielectric = dielectric_fresnel(h_v, 1.f, params.ior);
	let metallic = shlick(r0, dot(wi, wm));
	return lerp(f32x3(dielectric), metallic, params.metallic);
}

public BsdfEval eval_ggx(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	let n_l = brdf_cos(wi);
	let n_v = brdf_cos(wo);
	if (n_l <= 0.f || n_v <= 0.f)
		return { 0.f, 0.f, 0.f };

	let aniso = Aniso(params.roughness, params.anisotropic);

	let d = ggx_d(wm, aniso.ax, aniso.ay);
	let gl = ggx_g1(wi, wm, aniso.ax, aniso.ay);
	let gv = ggx_g1(wo, wm, aniso.ax, aniso.ay);
	let f = fresnel(params, wo, wm, wi);
	let eval = d * gl * gv * f / (4.f * n_l * n_v);

	var pdf = eval_ggx_v_aniso(wi, wm, wo, aniso.ax, aniso.ay);
	pdf.f_pdf *= 1.f / (4.f * abs(dot(wo, wm)));
	pdf.r_pdf *= 1.f / (4.f * abs(dot(wi, wm)));

	return { eval, pdf.r_pdf, pdf.f_pdf };
}

public BsdfSample sample_ggx(inout Rng rng, ShadingParams params, f32x3 wo) {
	let aniso = Aniso(params.roughness, params.anisotropic);

	let wm = sample_ggx_v_aniso(wo, aniso.ax, aniso.ay, rng.sample2());

	let wi = normalize(reflect(wm, wo));
	if (brdf_cos(wi) <= 0.f) {
		return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, true };
	}

	let f = fresnel(params, wo, wm, wi);
	let g1v = ggx_g1(wo, wm, aniso.ax, aniso.ay);
	let specular = g1v * f;

	var pdf = eval_ggx_v_aniso(wi, wm, wo, aniso.ax, aniso.ay);
	pdf.f_pdf *= 1.f / (4.f * abs(dot(wo, wm)));
	pdf.r_pdf *= 1.f / (4.f * abs(dot(wi, wm)));

	return { specular, wi, pdf.r_pdf, pdf.f_pdf, true };
}
