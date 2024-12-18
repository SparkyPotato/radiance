module pt;

import graph;
import graph.util;
import graph.util.rng;
import asset;
import passes.brdf;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	STex2D<f32x3> output;
	Rng rng;
	AS as;
	u32 samples;
};

[vk::push_constant]
PushConstants Constants;

RayDesc primary_ray(inout Rng rng, u32x2 pix) {
	let size = Constants.output.size();
	let uv = (f32x2(pix) + rng.sample2()) / f32x2(size);
	let clip = f32x2(uv.x, uv.y) * 2.f - 1.f;
	let cam = *Constants.camera;
	let origin = mul(cam.inv_view(), f32x4(0.f, 0.f, 0.f, 1.f)).xyz;
	let view_dir = normalize(mul(cam.inv_proj(), f32x4(clip.x, -clip.y, 1.f, 1.f)).xyz);
	let dir = mul(cam.inv_view(), f32x4(view_dir, 0.f)).xyz;
	return ray(origin, dir);
}

RayDesc ray(f32x3 origin, f32x3 wi) {
	RayDesc r;
	r.Origin = origin;
	r.Direction = wi;
	r.TMin = 1e-5f;
	r.TMax = 1e10f;
	return r;
}

bool is_unshadowed(RayDesc ray) {
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, ray);
	q.Proceed();
	return q.CommittedStatus() == COMMITTED_NOTHING;
}

struct Hit {
	f32x3 position;
	f32x3 normal;
	f32x3 base_color;
}

Optional<Hit> trace_ray(RayDesc ray) {
	RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
	q.TraceRayInline(Constants.as.get(), RAY_FLAG_NONE, 0xff, ray);
	while (q.Proceed());
	if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
		Hit h;

		let instance = Constants.instances[q.CommittedRayInstanceCustomIndex()];
		let tri = q.CommittedRayPrimitiveIndex();
		let iptr = (u32*)(instance.raw_mesh + instance.raw_vertex_count);
		let indices = u32x3(iptr[tri * 3], iptr[tri * 3 + 1], iptr[tri * 3 + 2]);
		let v0 = instance.raw_mesh[indices.x];
		let v1 = instance.raw_mesh[indices.y];
		let v2 = instance.raw_mesh[indices.z];
		let b = q.CommittedRayBarycentrics();
		let bary = f32x3(1.f - b.x - b.y, b.x, b.y);
		let mat = instance.material;

		let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
		let tmat = instance.transform.mat();
		let position = mul(tmat, f32x4(bary.x * v0.position + bary.y * v1.position + bary.z * v2.position, 1.f)).xyz;
		let normal = mul(tmat, f32x4(bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal, 0.f)).xyz;

		let bc = mat->base_color.get();
		let mr = mat->metallic_roughness.get();
		let nm = mat->normal.get();
		let em = mat->emissive.get();
		let white = f32x4(1.f);

		h.position = position;
		h.normal = normal;
		h.base_color = (bc.load(bc.pixel_of_uv(uv), white) * mat->base_color_factor).xyz;
		return h;
	} else
		return none;
}

f32x3 sample_sky() {
	return f32x3(0.43f, 0.61f, 0.92f) * 3.f;
}

[shader("pixel")]
void main(ScreenOutput input) {
	let pix = Constants.output.pixel_of_uv(input.uv);
	var rng = Constants.rng.init_at(pix);

	f32x3 ret = 0.f;
	for (int i = 0; i < 20; i++) {
		if (let hit = trace_ray(primary_ray(rng, pix))) {
			let brdf = Lambertian(hit.base_color);
			let sample = brdf.sample(rng, f32x3(0.f, 0.f, 0.f));
			let wi_w = mul(gen_basis(hit.normal), sample.sample.wi);
			if (is_unshadowed(ray(hit.position, wi_w))) {
				ret += sample.sample.color * sample_sky() / sample.pdf;
			}
		} else {
			ret += sample_sky();
		}
	}

	let prev = Constants.output.load(pix);
	let curr = ret / 20.f;
	let n = f32(Constants.samples);
	let n1 = n + 1.f;
	Constants.output.store(pix, prev * n / n1 + curr / n1);
}

f32x3x3 gen_basis(f32x3 z) {
	f32x3 other;
	if (z.x < 0.1f && z.y < 0.1f) {
		other = f32x3(0.f, -z.z, z.y);
	} else {
		other = f32x3(-z.y, z.x, 0.f);
	}
	other = normalize(other);
	let last = cross(other, z);
	// clang-format off
	return { 
		other.x, last.x, z.x,
		other.y, last.y, z.y,
		other.z, last.z, z.z
	};
	// clang-format on
}
