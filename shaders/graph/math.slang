implementing graph;

public static const f32 PI = 3.14159265359f;

public f32x4x4 quat_to_mat(f32x4 quat) {
	let x = quat.x;
	let y = quat.y;
	let z = quat.z;
	let w = quat.w;
	let x2 = x * x;
	let y2 = y * y;
	let z2 = z * z;

	// clang-format off
	return {
		1.f - 2.f * (y2 + z2), 2.f * (x * y - z * w), 2.f * (x * z + y * w), 0.f,
		2.f * (x * y + z * w), 1.f - 2.f * (x2 + z2), 2.f * (y * z - x * w), 0.f,
		2.f * (x * z - y * w), 2.f * (y * z + x * w), 1.f - 2.f * (x2 + y2), 0.f,
		0.f                  , 0.f                  , 0.f                  , 1.f
	};
	// clang-format on
}

public f32x4 axis_angle_to_quat(f32x3 axis, f32 angle) {
	let s = sin(angle * 0.5f);
	let c = cos(angle * 0.5f);
	return f32x4(axis.xyz * s, c);
}
