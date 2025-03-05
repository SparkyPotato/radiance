module common;

// References:
// A Scalable and Production Ready Sky and Atmosphere Rendering Technique
// Doesn't implement extinction froxels.
//
// https://sebh.github.io/publications/egsr2020.pdf
// https://www.shadertoy.com/view/slSXRW
// https://github.com/Sunset-Flock/Timberdoodle/

import graph;
import graph.util.color;

// TODO: Render in Rec2020.

// TODO: parameterize over these.
public static const f32 GROUND_RADIUS_MM = 6.36f;
public static const f32 ATMOSPHERE_RADIUS_MM = 6.46f;
public static const f32x3 GROUND_ALBEDO = f32x3(0.3f);

public static const f32x3 RAYLEIGH_SCATTERING_BASE = f32x3(5.802f, 13.558f, 33.1f);
public static const f32 RAYLEIGH_ABSORPTION_BASE = 0.0f;
public static const f32 MIE_SCATTERING_BASE = 3.996f;
public static const f32 MIE_ABSORPTION_BASE = 0.444f;
public static const f32x3 OZONE_ABSORPTION_BASE = f32x3(0.65f, 1.881f, 0.085f);

public f32 mie_phase(f32 cos) {
	let g = 0.8f;
	let scale = 3.f / (8.f * PI);

	let num = (1.f - g * g) * (1.f + cos * cos);
	let denom = (2.f + g * g) * pow((1.f + g * g - 2.f * g * cos), 1.5f);

	return scale * num / denom;
}

public f32 rayleigh_phase(f32 cos) {
	let k = 3.f / (16.f * PI);
	return k * (1.f + cos * cos);
}

public struct Scattering {
	public f32x3 rayleigh;
	public f32 mie;
	public f32x3 extinction;
}

public Scattering get_scattering(f32x3 pos) {
	let altitude_km = (length(pos) - GROUND_RADIUS_MM) * 1000.f;
	let rayleigh_density = exp(-altitude_km / 8.f);
	let mie_density = exp(-altitude_km / 1.2f);

	let rayleigh_scatter = RAYLEIGH_SCATTERING_BASE * rayleigh_density;
	let rayleigh_absorption = RAYLEIGH_ABSORPTION_BASE * rayleigh_density;

	let mie_scatter = MIE_SCATTERING_BASE * mie_density;
	let mie_absorption = MIE_ABSORPTION_BASE * mie_density;

	let ozone_absorption = OZONE_ABSORPTION_BASE * max(0.0, 1.0 - abs(altitude_km - 25.0) / 15.0);

	let extinction = rayleigh_scatter + rayleigh_absorption + mie_scatter + mie_absorption + ozone_absorption;

	return { rayleigh_scatter, mie_scatter, extinction };
}

public f32 safeacos(f32 x) {
	return acos(clamp(x, -1.f, 1.f));
}

// From https://gamedev.stackexchange.com/questions/96459/fast-ray-sphere-collision-code.
public f32 intersect_sphere(f32x3 ro, f32x3 rd, f32 rad) {
	let b = dot(ro, rd);
	let c = dot(ro, ro) - rad * rad;
	if (c > 0.0f && b > 0.0)
		return -1.0;
	let discr = b * b - c;
	if (discr < 0.0)
		return -1.0;
	// Special case: inside sphere, use far discriminant
	if (discr > b * b)
		return (-b + sqrt(discr));
	return -b - sqrt(discr);
}

public f32x2 lut_uv(f32x3 pos, f32x3 sun_dir) {
	let height = length(pos);
	let up = pos / height;
	let zenith_cos = dot(up, sun_dir);
	return f32x2(clamp(0.5f + 0.5f * zenith_cos, 0.f, 1.f),
				 clamp((height - GROUND_RADIUS_MM) / (ATMOSPHERE_RADIUS_MM - GROUND_RADIUS_MM), 0.f, 1.f));
}
