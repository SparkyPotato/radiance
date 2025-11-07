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

[shader("raygeneration")]
void main() {
	let pix = DispatchRaysIndex().xy;
	var rng = Constants.rng.init_at(pix);
	let ray = primary_ray(rng, pix);

	var p = HitPayload(rng);
	p.trace(ray);
}
