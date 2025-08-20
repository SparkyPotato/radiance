import graph;
import graph.util.color;

import asset;

public struct NodeData {
	public f32x3 aabb_min;
	public f32x3 aabb_max;
	public f32x4 position;	// variance is w
	public f32x3 power;
	public f32x3 normal_average;
	public f32x4 cone;

	public __init(NodeData data, Transform transform) {
		let mat = transform.mat();
		let rot = quat_to_mat(transform.rotation);
		let scale = max3(transform.scale.x, transform.scale.y, transform.scale.z);
		let aabb =
			transform_aabb(mat, Aabb((data.aabb_min + data.aabb_max) * 0.5f, (data.aabb_max - data.aabb_min) * 0.5f));
		this.aabb_min = aabb[0];
		this.aabb_max = aabb[1];
		this.position.xyz = mul(mat, f32x4(data.position.xyz, 1.f)).xyz;
		this.position.w = data.position.w * scale;
		this.power = data.power * scale;
		this.normal_average = mul(mat, f32x4(data.normal_average, 0.f)).xyz;
		this.cone = f32x4(mul(mat, f32x4(data.cone.xyz, 0.f)).xyz, data.cone.w);
	}

	public __init(f32x3 vertices[3], Material* material) {
		let e1 = vertices[1] - vertices[0];
		let e2 = vertices[2] - vertices[0];
		let area2 = cross(e1, e2);
		let normal = normalize(area2);
		this.aabb_min = min3(vertices[0], vertices[1], vertices[2]);
		this.aabb_max = max3(vertices[0], vertices[1], vertices[2]);
		this.position.xyz = (vertices[0] + vertices[1] + vertices[2]) / 3.f;
		let e1_l = length(e1);
		let e2_l = length(e2);
		this.position.w = (e1_l * e1_l + e2_l * e2_l - dot(e1, e2)) / 18.f;
		this.power = material->emissive_factor * length(area2) * 0.5f * PI;
		this.normal_average = normal * 0.5f;
		this.cone = f32x4(normal, 0.f);
	}

	public __init(NodeData left, NodeData right) {
		let w_left = luminance_rec709(left.power) / luminance_rec709(left.power + right.power);
		let w_right = 1.f - w_left;
		this.aabb_min = min(left.aabb_min, right.aabb_min);
		this.aabb_max = max(left.aabb_max, right.aabb_max);
		this.position = left.position * w_left + right.position * w_right;
		let pos_diff = left.position.xyz - right.position.xyz;
		this.position.w += w_left * w_right * dot(pos_diff, pos_diff);
		this.power = left.power + right.power;
		this.normal_average = left.normal_average * w_left + right.normal_average * w_right;
		this.cone = cone_union(left.cone, right.cone);
	}

	public static NodeData wave_shuffle(NodeData data, u32 wid) {
		NodeData ret;
		ret.aabb_min = WaveShuffle(data.aabb_min, wid);
		ret.aabb_max = WaveShuffle(data.aabb_max, wid);
		ret.position = WaveShuffle(data.position, wid);
		ret.power = WaveShuffle(data.power, wid);
		ret.normal_average = WaveShuffle(data.normal_average, wid);
		ret.cone = WaveShuffle(data.cone, wid);
		return ret;
	}
}

f32x4 cone_union(f32x4 a, f32x4 b) {
	let a_len = length(a.xyz);
	let b_len = length(b.xyz);
	if (a_len < 0.00001f || b_len < 0.00001f)
		return f32x4(a.xyz, PI);

	let theta_d = acos(clamp(dot(a.xyz, b.xyz), -1.f, 1.f));
	if (min(theta_d + b.w, PI) <= a.w)
		return a;
	if (min(theta_d + a.w, PI) <= b.w)
		return b;
	let theta_o = (a.w + b.w + theta_d) * 0.5f;
	if (theta_o >= PI)
		return f32x4(a.xyz, PI);

	let theta_r = theta_o - a.w;
	let w_r = cross(a.xyz, b.xyz);
	if (dot(w_r, w_r) < 0.00001f)
		return f32x4(a.xyz, PI);

	let axis = mul(quat_to_mat(axis_angle_to_quat(w_r, theta_r)), f32x4(a.xyz, 0.f)).xyz;
	return f32x4(axis, theta_o);
}

public struct LightBlas {
	public LightTreeNode* root;
	public Vertex* vertices;
	public u32x3* indices;
	public Material* material;
	public NodeData data;
}

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
