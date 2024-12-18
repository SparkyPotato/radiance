module pt;

import graph;
import graph.util;
import graph.util.rng;
import asset;
import passes.brdf;
import passes.visbuffer;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	STex2D<f32x3> output;
	Rng rng;
	AS as;
};

[vk::push_constant]
PushConstants Constants;

struct MaterialData {
	f32x3 base_color;
	f32 roughness;
	f32 metallic;
	f32x3 emissive;
	f32x3 normal;
	f32x3x3 basis;
	f32x3x3 inv_basis;

	__init(DecodedTri tri) {
		let uv = tri.uv();
		let mat = tri.instance.material;
		let bc = mat->base_color.get();
		let mr = mat->metallic_roughness.get();
		let em = mat->emissive.get();
		let white = f32x4(1.f, 1.f, 1.f, 1.f);
		this.base_color = (bc.load(bc.pixel_of_uv(uv), white) * mat->base_color_factor).xyz;
		let v = mr.load(mr.pixel_of_uv(uv), 1.f);
		this.metallic = v.x * mat->metallic_factor;
		this.roughness = v.y * mat->roughness_factor;
		this.emissive = em.load(em.pixel_of_uv(uv), white).xyz * mat->emissive_factor;
		this.basis = tri.tbn_basis();
		this.inv_basis = transpose(this.basis);
		let nm = tri.instance.material->normal.get();
		let z = f32x4(0.f, 0.f, 1.f, 0.f);
		this.normal = nm.load(nm.pixel_of_uv(tri.uv()), z).xyz;
	}
}

f32x3 sample_sky() {
	return f32x3(0.43f, 0.61f, 0.92f) * 3.f;
}

bool is_shadowed(f32x3 origin, f32x3 wi) {
	RayQuery<RAY_FLAG_NONE> q;
	RayDesc r;
	r.Origin = origin;
	r.Direction = wi;
	r.TMin = 1e-5f;
	r.TMax = 1e10f;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, r);
	while (q.Proceed());
	return q.CommittedStatus() != COMMITTED_NOTHING;
}

f32x3 run_sample(inout Rng rng, ScreenOutput input) {
	// let tri = DecodedTri(Constants.instances, Constants.camera[0], input.uv, Constants.read.size(), p);
	// let data = MaterialData(tri);
	// let brdf = Lambertian(data.base_color);
	// let samp = brdf.sample(rng, f32x3(0.f, 0.f, 0.f));
	// let wi_w = mul(data.basis, samp.sample.wi);
	// let unshadowed = is_shadowed(pr, pwi);

	return f32x3(0.f, 0.f, 0.f);
}

[shader("pixel")]
void main(ScreenOutput input) {
	let pix = Constants.output.pixel_of_uv(input.uv);
	var rng = Constants.rng.init_at(pix);
	let size = Constants.output.size();

	let uv = (f32x2(pix) + rng.sample2()) / f32x2(size);
	let clip = f32x2(uv.x, -uv.y) * 2.f - 1.f;
	let cam = *Constants.camera;
	let target = normalize(mul(cam.inv_proj(), f32x4(clip, 1.f, 1.f)).xyz);
	let origin = mul(cam.inv_view(), f32x4(0.f, 0.f, 0.f, 1.f)).xyz;
	let dir = mul(cam.inv_view(), f32x4(target, 0.f)).xyz;

	let ret = is_shadowed(origin, dir);
	Constants.output.store(pix, ret);
}
