module gen;

import graph;
import graph.util.color;
import graph.util.rng;
import asset;
import passes.bsdf;
import common;

Ray primary_ray(inout Rng rng, u32x2 pix) {
	let size = Constants.output.size();
	let uv = (f32x2(pix) + rng.sample2()) / f32x2(size);
	let clip = f32x2(uv.x, uv.y) * 2.f - 1.f;
	let cam = *Constants.camera;
	let origin = mul(cam.inv_view(), f32x4(0.f, 0.f, 0.f, 1.f)).xyz;
	let view_dir = normalize(mul(cam.inv_proj(), f32x4(clip.x, -clip.y, 0.f, 1.f)).xyz);
	let dir = mul(cam.inv_view(), f32x4(view_dir, 0.f)).xyz;
	return Ray(origin, dir);
}

f32x3 li(inout Rng rng, Ray ray) {
	HitPayload p;
	p.rng = rng;
	p.ray = ray;
	p.L = f32x3(0.f);
	p.p_bounce = 0.f;
	p.specular = true;
	p.b = f32x3(1.f);
	p.prev_hit_norm = f32x3(0.f);

	for (int bounces = 0; bounces < 2; bounces++) {
		p.ray.trace(RAY_FLAG_FORCE_OPAQUE, 0, p);
		if (!p.hit)
			break;

		if (bounces > 1) {
			let q = max(0.05f, 1.f - luminance(p.b));
			if (p.rng.sample() < q)
				break;
			p.b /= 1.f - q;
		}
	}

	return p.L;
}

[shader("raygeneration")]
void main() {
	let pix = DispatchRaysIndex().xy;
	var rng = Constants.rng.init_at(pix);

	let r = primary_ray(rng, pix);
	var ret = li(rng, r);
	if (any(isnan(ret) || isinf(ret)))
		ret = f32x3(1.f, 0.f, 1.f) * 1e7f;

	let n = Constants.samples;
	if (n == 0) {
		Constants.output.store(pix, ret);
	} else {
		let prev = Constants.output.load(pix);
		Constants.output.store(pix, lerp(prev, ret, 1.f / (f32(n) + 1.f)));
	}
}
