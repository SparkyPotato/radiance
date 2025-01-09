import graph;
import common;

[shader("miss")]
void main(inout HitPayload p) {
	p.hit = false;

	let le = Constants.sky.sample_primary(p.ray.origin, p.ray.dir);
	if (p.bounces == 0 || p.specular) {
		p.L += p.b * le;
	} else {
		// MIS for the sky light.
		let p_light = light_sample_pdf() / (4.f * PI);
		let w = pow_heuristic(1, p.p_bounce, 1, p_light);
		p.L += p.b * w * le;
	}
}
