import graph;
import graph.util.color;
import graph.util.rng;
import passes.bsdf;
import common;

[shader("miss")]
void main(inout HitPayload p) {
	p.hit = false;

	let le = rec709_to_rec2020(Constants.sky.sample_primary(p.ray.origin, p.ray.dir));
	f32 w = 1.f;
	if (!p.specular) {
		// MIS for the sky light.
		let p_light = light_sample_pdf() * dot(p.ray.dir, p.prev_hit_norm) / PI;
		w = pow_heuristic_1(p.p_bounce, p_light);
	}
	p.L += p.b * w * le;
}
