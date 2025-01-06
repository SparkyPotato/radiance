module eval;

import graph;
import graph.util;
import passes.sky.common;

struct PushConstants {
	Tex2D<f32x3> sun_transmittance;
	Tex2D<f32x3> sun_scattering;
	Sampler sampler;
	f32x3 cam_pos;
	f32x3 sun_dir;
}

[vk::push_constant]
PushConstants Constants;

static const u32 STEPS = 32;

f32x3 raymarch(f32x3 pos, f32x3 dir, f32x3 sun_dir, f32 t_max) {
	let cos = dot(dir, sun_dir);
	let mie = mie_phase(cos);
	let rayleigh = rayleigh_phase(-cos);

	var lum = f32x3(0.f);
	var trans = f32x3(1.f);
	var t = 0.f;
	for (int i = 0; i < STEPS; i++) {
		let new_t = ((f32(i) + 0.3f) / f32(STEPS)) * t_max;
		let dt = new_t - t;
		t = new_t;
		let new_pos = pos + dir * t;

		let s = get_scattering(new_pos);
		let sample_trans = exp(-dt * s.extinction);
		let sun_trans = Constants.sun_transmittance.sample(Constants.sampler, map_to_lut_uv(new_pos, sun_dir));
		let ms = Constants.sun_scattering.sample(Constants.sampler, map_to_lut_uv(new_pos, sun_dir));

		let rayleigh_inscatter = s.rayleigh * (rayleigh * sun_trans + ms);
		let mie_inscatter = s.mie * (mie * sun_trans + ms);
		let inscatter = rayleigh_inscatter + mie_inscatter;

		let integral = (inscatter - inscatter * sample_trans) / s.extinction;
		lum += integral * trans;
		trans *= sample_trans;
	}
	return lum;
}

[shader("pixel")]
f32x3 main(ScreenOutput input) : SV_Target0 {
	let uv = input.uv;
	let azimuth = (uv.x - 0.5f) * 2.f * PI;
	f32 adj_v;
	if (uv.y < 0.5f) {
		let coord = 1.f - 2.f * uv.y;
		adj_v = -coord * coord;
	} else {
		let coord = uv.y * 2.f - 1.f;
		adj_v = coord * coord;
	}

	let pos = f32x3(Constants.cam_pos.xy / 1000000.f, Constants.cam_pos.z / 1000000.f + GROUND_RADIUS_MM);
	let height = length(pos);
	let up = pos / height;
	let horiz = safeacos(sqrt(height * height - GROUND_RADIUS_MM * GROUND_RADIUS_MM) / height) - PI / 2.f;
	let alt = adj_v * PI / 2.f - horiz;

	let cos_alt = cos(alt);
	let dir = f32x3(cos_alt * sin(azimuth), -cos_alt * cos(azimuth), sin(alt));

	let sun_alt = PI / 2.f - acos(dot(Constants.sun_dir, up));
	let sun_dir = f32x3(0.f, -cos(sun_alt), sin(sun_alt));

	let atmo_dist = intersect_sphere(pos, dir, ATMOSPHERE_RADIUS_MM);
	let ground_dist = intersect_sphere(pos, dir, GROUND_RADIUS_MM);
	let t_max = (ground_dist < 0.f) ? atmo_dist : ground_dist;
	return raymarch(pos, dir, sun_dir, t_max);
}
