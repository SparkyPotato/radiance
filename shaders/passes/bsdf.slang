module brdf;

import graph;
import graph.util.rng;
import passes.bsdf.common;
import passes.bsdf.anisotropic;
import passes.bsdf.diffuse;
import passes.bsdf.sheen;
import passes.bsdf.ggx;

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
		this.specular = 0.f;
		this.diffuse = 1.f;
		this.clearcoat = 0.f;
		this.spec_trans = 0.f;
	}
}

f32 thin_trans_roughness(f32 ior, f32 roughness) {
	return saturate((0.65f * ior - 0.35f) * roughness);
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
	// let diffuse_weight = (1.f - metallic) * (1.f - spec_trans);
	let diffuse_weight = 1.f;
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
	if (false) {
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
