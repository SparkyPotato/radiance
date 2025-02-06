module scattering;

import graph;
import graph.util;
import passes.sky.common;

struct PushConstants {
	Tex2D<f32x3> sun_transmittance;
	Sampler sampler;
}

[vk::push_constant]
PushConstants Constants;

static const u32 STEPS = 20;
static const u32 SQRT_SAMPLES = 8;

struct Scatter {
	f32x3 lum;
	f32x3 f_ms;
}

f32x3 spherical_dir(f32 theta, f32 phi) {
	let sin_phi = sin(phi);
	let cos_phi = cos(phi);
	let sin_theta = sin(theta);
	let cos_theta = cos(theta);
	return f32x3(sin_phi * sin_theta, sin_phi * cos_theta, cos_phi);
}

Scatter sun_scattering(f32x3 pos, f32x3 sun_dir) {
	let inv_samples = 1.f / f32(SQRT_SAMPLES * SQRT_SAMPLES);

	var lum_tot = f32x3(0.f);
	var f_ms = f32x3(0.f);
	for (int i = 0; i < SQRT_SAMPLES; i++) {
		for (int j = 0; j < SQRT_SAMPLES; j++) {
			let theta = PI * (f32(i) + 0.5f) / f32(SQRT_SAMPLES);
			let phi = safeacos(1.f - 2.f * (f32(j) + 0.5f) / f32(SQRT_SAMPLES));
			let dir = spherical_dir(theta, phi);

			let atmo_dist = intersect_sphere(pos, dir, ATMOSPHERE_RADIUS_MM);
			let ground_dist = intersect_sphere(pos, dir, GROUND_RADIUS_MM);
			let t_max = (ground_dist > 0.f) ? ground_dist : atmo_dist;

			let cos = dot(dir, sun_dir);
			let mie = mie_phase(cos);
			let rayleigh = rayleigh_phase(-cos);

			var lum = f32x3(0.f);
			var lum_f = f32x3(0.f);
			var trans = f32x3(1.f);
			var t = 0.f;
			for (int k = 0; k < STEPS; k++) {
				let new_t = ((f32(k) + 0.3f) / f32(STEPS)) * t_max;
				let dt = new_t - t;
				t = new_t;
				let new_pos = pos + dir * t;

				let s = get_scattering(new_pos);
				let sample_trans = exp(-dt * s.extinction);

				let scatter_no_phase = s.rayleigh + s.mie;
				let scatter_f = (scatter_no_phase - scatter_no_phase * sample_trans) / s.extinction;
				lum_f += trans * scatter_f;

				let sun_trans =
					Constants.sun_transmittance.sample_mip(Constants.sampler, lut_uv(new_pos, sun_dir), 0.f);

				let rayleigh_in = s.rayleigh * rayleigh;
				let mie_in = s.mie * mie;
				let inscattering = (rayleigh_in + mie_in) * sun_trans;

				let integrated = (inscattering - inscattering * sample_trans) / s.extinction;
				lum += integrated * trans;
				trans *= sample_trans;
			}

			if (ground_dist > 0.f) {
				if (dot(pos, sun_dir) > 0.f) {
					let hit = normalize(pos + ground_dist * dir) * GROUND_RADIUS_MM;
					lum += trans * GROUND_ALBEDO *
						   Constants.sun_transmittance.sample_mip(Constants.sampler, lut_uv(hit, sun_dir), 0.f);
				}
			}

			lum_tot += lum * inv_samples;
			f_ms += lum_f * inv_samples;
		}
	}

	return { lum_tot, f_ms };
}

[shader("pixel")]
f32x3 main(ScreenOutput input) : SV_Target0 {
	let uv = input.uv;
	let sun_cos = 2.f * uv.x - 1.f;
	let sun_theta = safeacos(sun_cos);
	let height = lerp(GROUND_RADIUS_MM, ATMOSPHERE_RADIUS_MM, uv.y);

	let dir = normalize(f32x3(0.f, -sin(sun_theta), sun_cos));
	let s = sun_scattering(f32x3(0.f, 0.f, height), dir);
	return s.lum / (1.f - s.f_ms);
}
