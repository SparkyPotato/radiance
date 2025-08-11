import graph;

public struct SgLight {
	public f32x3 position;
	public f32 variance;
	public f32x3 intensity;
	public f32x3 axis;
}

public struct LightTreeNode {
	public SgLight light;
	// LSB is if is inner node.
	public u32 L;
	// LSB is if is inner node.
	public u32 R;
}

static const f32 SG_LIGHT_MAX_SHARPNESS = 0x1.0p41;

f32 expm1_over_x(f32 x) {
	let u = exp(x);
	if (u == 1.f) {
		return 1.f;
	}
	let y = u - 1.f;
	if (abs(x) < 1.f) {
		return y / log(u);
	}
	return y / x;
}

f32 sg_integral(f32 sharpness) {
	return 4.f * PI * expm1_over_x(-2.f * sharpness);
}

public f32x3 power_to_intensity(f32x3 power, f32 sharpness) {
	return power / (2.f * PI * sg_integral(sharpness));
}

public f32x4 average_direction_to_sg_axis(f32x3 average) {
	let length = clamp(length(average), 0.f, 1.f);
	let length2 = length * length;
	let length3 = length2 * length;
	let sharpness = min((3.f * length - length3) / (1.f - length2), SG_LIGHT_MAX_SHARPNESS);
	let axis = length > 0.000001f ? average / length : f32x3(0.f);
	return f32x4(axis, sharpness);
}
