module transmittance;

import graph;
import graph.util;
import passes.sky.common;

static const u32 STEPS = 40;

f32x3 sun_transmittance(f32x3 pos, f32x3 sun_dir) {
	if (intersect_sphere(pos, sun_dir, GROUND_RADIUS_MM) > 0.f)
		return f32x3(0.f);

	let atmo_dist = intersect_sphere(pos, sun_dir, ATMOSPHERE_RADIUS_MM);

	var t = 0.f;
	var trans = f32x3(1.f);
	for (int i = 0; i < STEPS; i++) {
		let new_t = ((f32(i) + 0.3f) / f32(STEPS)) * atmo_dist;
		let dt = new_t - t;
		t = new_t;
		let new_pos = pos + sun_dir * t;

		trans *= exp(-dt * get_scattering(new_pos).extinction);
	}
	return trans;
}

[shader("pixel")]
f32x3 main(ScreenOutput input) : SV_Target0 {
	let uv = input.uv;
	let sun_cos = 2.f * uv.x - 1.f;
	let sun_theta = safeacos(sun_cos);
	let height = lerp(GROUND_RADIUS_MM, ATMOSPHERE_RADIUS_MM, uv.y);

	let pos = f32x3(0.f, 0.f, height);
	let dir = normalize(f32x3(0.f, -sin(sun_theta), sun_cos));
	return sun_transmittance(pos, dir);
}
