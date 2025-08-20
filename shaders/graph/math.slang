implementing graph;

public static const f32 PI = 3.14159265359f;
public static const f32 FLT_MAX = 3.402823466e+38F;

public struct Aabb {
	public f32x3 center;
	public f32x3 half_extent;
}

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

public vector<T, N> min8<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> p0, vector<T, N> p1, vector<T, N> p2,
																	  vector<T, N> p3, vector<T, N> p4, vector<T, N> p5,
																	  vector<T, N> p6, vector<T, N> p7) {
	return min(p0, min(p1, min(p2, min(p3, min(p4, min(p5, min(p6, p7)))))));
}

public vector<T, N> max8<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> p0, vector<T, N> p1, vector<T, N> p2,
																	  vector<T, N> p3, vector<T, N> p4, vector<T, N> p5,
																	  vector<T, N> p6, vector<T, N> p7) {
	return max(p0, max(p1, max(p2, max(p3, max(p4, max(p5, max(p6, p7)))))));
}

public f32x3[2] transform_aabb(f32x4x4 transform, Aabb aabb) {
	let extent = aabb.half_extent * 2.f;
	let sx = mul(transform, f32x4(extent.x, 0.f, 0.f, 0.f));
	let sy = mul(transform, f32x4(0.f, extent.y, 0.f, 0.f));
	let sz = mul(transform, f32x4(0.f, 0.f, extent.z, 0.f));

	let p0 = mul(transform, f32x4(aabb.center - aabb.half_extent, 1.f));
	let p1 = p0 + sz;
	let p2 = p0 + sy;
	let p3 = p2 + sz;
	let p4 = p0 + sx;
	let p5 = p4 + sz;
	let p6 = p4 + sy;
	let p7 = p6 + sz;

	let min = min8(p0, p1, p2, p3, p4, p5, p6, p7).xyz;
	let max = max8(p0, p1, p2, p3, p4, p5, p6, p7).xyz;
	return { min, max };
}
