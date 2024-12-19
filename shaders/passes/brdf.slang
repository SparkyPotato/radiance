module brdf;

import graph;
import graph.util.rng;

public struct BsdfSample {
	public f32x3 L;
	public f32x3 wi;
	public f32 r_pdf;
	public f32 f_pdf;
	public bool is_specular;
}

public struct BsdfEval {
	public f32x3 L;
	public f32 r_pdf;
	public f32 f_pdf;
}

public struct ShadingParams {
	public f32x3 base_color;
	public f32x3 sheen_tint;
	public f32x3 spec_tint;
	public f32 sheen;
	public f32 metallic;
	public f32 roughness;
	public f32 spec_trans;
	public f32 clearcoat;
	public f32 rel_ior;
	public f32 ior;
	public f32 anisotropic;
	public f32 flatness;
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

f32x3 shlick(f32x3 r0, f32 rad) {
	let exp = pow(1.f - rad, 5.f);
	return r0 + (f32x3(1.f) - r0) * exp;
}

f32 shlick(f32 r0, f32 rad) {
	return lerp(1.f, shlick_weight(rad), r0);
}

f32 shlick_weight(f32 u) {
	let m = saturate(1.f - u);
	let m2 = m * m;
	return m * m2 * m2;
}

f32 shlick_r0_from_rel_ior(f32 eta) {
	let em = eta - 1.f;
	let ep = eta + 1.f;
	return (em * em) / (ep * ep);
}

f32 dielectric_fresnel(f32 cos_i, f32 ni, f32 nt) {
	cos_i = clamp(cos_i, -1.f, 1.f);
	if (cos_i < 0.f) {
		let t = ni;
		ni = nt;
		nt = t;
		cos_i - cos_i;
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

f32 ggx_g1(f32x3 w, f32x3 wm, f32 ax, f32 ay) {
	let h_w = dot(w, wm);
	if (h_w <= 0.f)
		return 0.f;

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

struct GgxVAnisoPdf {
	f32 r_pdf;
	f32 f_pdf;

	__init(f32x3 wi, f32x3 wm, f32x3 wo, f32 ax, f32 ay) {
		let d = ggx_d(wm, ax, ay);

		let abs_n_l = brdf_abscos(wi);
		let abs_h_l = abs(dot(wm, wi));
		let g1v = ggx_g1(wo, wm, ax, ay);
		this.f_pdf = g1v * abs_h_l * d / abs_n_l;

		let abs_n_v = brdf_abscos(wo);
		let abs_h_v = abs(dot(wm, wo));
		let g1l = ggx_g1(wi, wm, ax, ay);
		this.r_pdf = g1l * abs_h_v * d / abs_n_v;
	}
}

f32x3 sample_ggx_v_aniso(f32x3 wo, f32 ax, f32 ay, f32x2 u) {
	let v = normalize(f32x3(wo.x * ax, wo.y * ay, wo.z));

	let t1 = (v.y < 0.9999f) ? normalize(cross(v, f32x3(0.f, 1.f, 0.f))) : f32x3(1.f, 0.f, 0.f);
	let t2 = cross(t1, v);

	let a = 1.f / (1.f + v.z);
	let r = sqrt(u.x);
	let phi = (u.y < a) ? (u.y / a) * PI : PI + (u.y - a) / (1.f - a) * PI;
	let p1 = r * cos(phi);
	let p2 = r * sin(phi) * ((u.y < a) ? 1.f : v.z);

	let n = p1 * t1 + p2 * t2 + sqrt(max(0.f, 1.f - p1 * p1 - p2 * p2));
	return normalize(f32x3(ax * n.x, ay * n.y, n.z));
}

struct LobePdfs {
	f32 specular;
	f32 diffuse;
	f32 clearcoat;
	f32 spec_trans;

	__init(ShadingParams params) {
		let metallic = params.metallic;
		let specular = (1.f - params.metallic) * params.spec_trans;
		let dielectric = (1.f - params.spec_trans) * (1.f - params.metallic);

		let spec_w = metallic + dielectric;
		let diffuse_w = dielectric;
		let trans_w = specular;
		let clearcoat_w = 1.f * saturate(params.clearcoat);

		let norm = 1.f / (spec_w + trans_w + diffuse_w + clearcoat_w);
		this.specular = spec_w * norm;
		this.diffuse = diffuse_w * norm;
		this.clearcoat = clearcoat_w * norm;
		this.spec_trans = trans_w * norm;
	}
}

f32 thin_trans_roughness(f32 ior, f32 roughness) {
	return saturate((0.65f * ior - 0.35f) * roughness);
}

struct Aniso {
	f32 ax;
	f32 ay;

	__init(f32 roughness, f32 anisotropic) {
		let aspect = sqrt(1.f - 0.9f * anisotropic);
		let alpha = roughness * roughness;
		this.ax = max(0.001f, alpha / aspect);
		this.ay = max(0.001f, alpha * aspect);
	}
}

f32x3 tint(f32x3 base_color) {
	let lum = dot(f32x3(0.3f, 0.6f, 1.f), base_color);
	return (lum > 0.f) ? base_color / lum : f32x3(1.f);
}

f32 gtr1(f32 abs_h_l, f32 a) {
	if (a >= 1.f)
		return 1.f / PI;
	let a2 = a * a;
	return (a2 - 1.f) / (PI * log2(a2) * (1.f + (a2 - 1.f) * abs_h_l * abs_h_l));
}

struct Clearcoat {
	f32 eval;
	f32 r_pdf;
	f32 f_pdf;

	__init(f32 clearcoat, f32 alpha, f32x3 wo, f32x3 wm, f32x3 wi) {
		if (clearcoat <= 0.f) {
			this.eval = 0.f;
			this.r_pdf = 0.f;
			this.f_pdf = 0.f;
			return;
		}

		let abs_n_h = brdf_abscos(wm);
		let abs_n_l = brdf_abscos(wi);
		let abs_n_v = brdf_abscos(wo);
		let h_l = dot(wm, wi);

		let d = gtr1(abs_n_h, lerp(0.1f, 0.001, alpha));
		let f = shlick(0.04f, h_l);
		let gl = ggx_g1(wi, 0.25f);
		let gv = ggx_g1(wo, 0.25f);

		this.r_pdf = d / (4.f * abs(dot(wi, wm)));
		this.f_pdf = d / (4.f * abs(dot(wo, wm)));
		this.eval = 0.25f * clearcoat * d * f * gl * gv;
	}
}

f32x3 eval_sheen(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	if (params.sheen <= 0.f)
		return f32x3(0.f);

	let h_l = abs(dot(wm, wi));
	let tint = tint(params.base_color);
	return params.sheen * lerp(f32x3(1.f), tint, params.sheen_tint) * shlick_weight(h_l);
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

BsdfEval eval_ggx(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
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

	var pdf = GgxVAnisoPdf(wi, wm, wo, aniso.ax, aniso.ay);
	pdf.f_pdf *= 1.f / (4.f * abs(dot(wo, wm)));
	pdf.r_pdf *= 1.f / (4.f * abs(dot(wi, wm)));

	return { eval, pdf.r_pdf, pdf.f_pdf };
}

BsdfSample sample_ggx(inout Rng rng, ShadingParams params, f32x3 wo) {
	let aniso = Aniso(params.roughness, params.anisotropic);

	let wm = sample_ggx_v_aniso(wo, aniso.ax, aniso.ay, rng.sample2());

	let wi = normalize(reflect(wm, wo));
	if (brdf_cos(wi) <= 0.f) {
		return { f32x3(0.f), f32x3(0.f), 0.f, 0.f, true };
	}

	let f = fresnel(params, wo, wm, wi);
	let g1v = ggx_g1(wo, wm, aniso.ax, aniso.ay);
	let specular = g1v * f;

	var pdf = GgxVAnisoPdf(wi, wm, wo, aniso.ax, aniso.ay);
	pdf.f_pdf *= 1.f / (4.f * abs(dot(wo, wm)));
	pdf.r_pdf *= 1.f / (4.f * abs(dot(wi, wm)));

	return { specular, wi, pdf.r_pdf, pdf.f_pdf, true };
}

f32 eval_retro_diffuse(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	let n_l = brdf_abscos(wi);
	let n_v = brdf_abscos(wo);

	let roughness = params.roughness * params.roughness;
	let rr = 0.5f + 2.f * n_l * n_l * roughness;
	let fl = shlick_weight(n_l);
	let fv = shlick_weight(n_v);

	return rr * (fl + fv + fl * fv * (rr - 1.f));
}

f32 eval_diffuse(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi, bool thin) {
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

BsdfSample sample_diffuse(inout Rng rng, ShadingParams params, f32x3 wo, bool thin) {
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

// TODO: Spec trans, clearcoat.

public BsdfEval eval_disney_bsdf(ShadingParams params, f32x3 wo, f32x3 wi, bool thin) {
	let wm = normalize(wo + wi);
	let n_v = brdf_cos(wo);
	let n_l = brdf_cos(wi);

	var L = f32x3(0.f);
	var r_pdf = 0.f;
	var f_pdf = 0.f;

	let pdf = LobePdfs(params);
	let aniso = Aniso(params.roughness, params.anisotropic);
	let metallic = params.metallic;
	let spec_trans = params.spec_trans;
	let diffuse_weight = (1.f - metallic) * (1.f - spec_trans);
	let trans_weight = (1.f - metallic) * spec_trans;
	let upper_hemi = n_l > 0.f && n_v > 0.f;

	// diffuse
	if (diffuse_weight > 0.f) {
		let fd_pdf = brdf_abscos(wi);
		let rd_pdf = brdf_abscos(wo);
		let diffuse = eval_diffuse(params, wo, wm, wi, thin);
		let sheen = eval_sheen(params, wo, wm, wi);

		L += diffuse_weight * (diffuse * params.base_color + sheen);
		r_pdf += pdf.diffuse * rd_pdf;
		f_pdf += pdf.diffuse * fd_pdf;
	}

	// specular
	if (upper_hemi) {
		let s = eval_ggx(params, wo, wm, wi);
		L += s.L;
		r_pdf += pdf.specular * s.r_pdf / (4.f * abs(dot(wo, wm)));
		f_pdf += pdf.specular * s.f_pdf / (4.f * abs(dot(wi, wm)));
	}

	return { L * abs(n_l), r_pdf, f_pdf };
}

public BsdfSample sample_disney_bsdf(inout Rng rng, ShadingParams params, f32x3 wo, bool thin) {
	let pdf = LobePdfs(params);

	BsdfSample ret;
	var p_lobe = 0.f;
	let p = rng.sample();
	if (p <= pdf.specular) {
		ret = sample_ggx(rng, params, wo);
		p_lobe = pdf.specular;
	} else if (p <= (pdf.specular + pdf.clearcoat)) {
		p_lobe = pdf.clearcoat;
	} else if (p <= (pdf.specular + pdf.clearcoat + pdf.diffuse)) {
		ret = sample_diffuse(rng, params, wo, thin);
		p_lobe = pdf.diffuse;
	} else if (pdf.spec_trans >= 0.f) {
		p_lobe = pdf.spec_trans;
	} else {
		ret.L = f32x3(1000000.f, 0.f, 1000000.f);
		ret.r_pdf = 1.f;
		ret.f_pdf = 1.f;
	}

	if (p_lobe > 0.f) {
		ret.L /= p_lobe;
		ret.r_pdf *= p_lobe;
		ret.f_pdf *= p_lobe;
	}
	return ret;
}
