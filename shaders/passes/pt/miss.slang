import graph;
import passes.bsdf;
import common;

[shader("miss")]
void main(inout HitPayload p) {
	p.hit = false;

	let le = Constants.sky.sample_primary(p.ray.origin, p.ray.dir);
	f32 w = 1.f;
	// let le = f32x3(10.f);
	if (!p.specular) {
		// MIS for the sky light.
		let p_light = light_sample_pdf() * dot(p.ray.dir, p.prev_hit_norm) / PI;
		w = pow_heuristic(1, p.p_bounce, 1, p_light);
	}
	p.L += p.b * w * le;
}
