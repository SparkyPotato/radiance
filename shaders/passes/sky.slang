module sky;

import graph;
import passes.sky.common;

// TODO: reduce the size of this, as it goes in push constants.
public struct SkySampler {
	Tex2D<f32x3> sky;
	Sampler sampler;
	f32x3 sun_dir;
	f32x3 sun_radiance;

	f32x3 sun_disk(f32x3 dir) {
		let sun_cos = cos(radians(0.5f));
		let cos = dot(dir, this.sun_dir);
		if (cos >= sun_cos)
			return this.sun_radiance;

		let offset = sun_cos - cos;
		let gaussian = exp(-offset * 50000.f) / 2.f;
		let inv = 1.f / (0.02f + offset * 300.f) * 0.01f;
		return (gaussian + inv) * this.sun_radiance;
	}

	public f32x3 sample_primary(f32x3 pos, f32x3 dir) {
		let pos = f32x3(pos.xy / 1000000.f, pos.z / 1000000.f + GROUND_RADIUS_MM);
		let li = this.sample(pos, dir);
		var sun = smoothstep(0.002f, this.sun_radiance, this.sun_disk(dir));
		if (any(sun > 0.f)) {
			if (intersect_sphere(pos, dir, GROUND_RADIUS_MM) >= 0.f) {
				sun = f32x3(0.f);
			}
		}

		return li + sun;
	}

	public f32x3 sample(f32x3 pos, f32x3 dir) {
		let pos = f32x3(pos.xy / 1000000.f, pos.z / 1000000.f + GROUND_RADIUS_MM);
		let height = length(pos);
		let up = pos / height;

		let horizon = safeacos(sqrt(height * height - GROUND_RADIUS_MM * GROUND_RADIUS_MM) / height);
		let altitude = horizon - acos(dot(dir, up));
		f32 azimuth;
		if (abs(altitude) > (PI / 2.f - 0.0001f)) {
			azimuth = 0.f;
		} else {
			let right = cross(this.sun_dir, up);
			let forward = cross(up, right);
			let projected = normalize(dir - up * dot(dir, up));
			let sin = dot(projected, right);
			let cos = dot(projected, forward);
			azimuth = atan2(sin, cos) + PI;
		}

		let v = 0.5f + 0.5f * sign(altitude) * sqrt(abs(altitude) * 2.f / PI);
		let uv = f32x2(azimuth / (2.f * PI), v);
		return this.sky.sample(this.sampler, uv) * this.sun_radiance;
	}
}
