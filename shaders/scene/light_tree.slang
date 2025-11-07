import graph;
import graph.util.color;
import graph.util.rng;

import asset;

public struct LightSamplePoint {
	public Sampler sampler;
	public f32x3 position;
	public f32x3 normal;
	public f32x3 base_color;
	public f32 roughness;
	public f32 metallic;
}

public struct LightSample {
	public f32x3 L;
	public f32x3 wi_world;
	public f32 t;
	public f32 pdf;
}

public struct LightTree {
	LightTreeNode* tlas;
	BlasInstance* blases;

	public Optional<LightSample> sample(inout Rng rng, LightSamplePoint point) {
		return none;
		var p = rng.sample();
		var index = 0;
		var pdf = 1.f;
		for (u32 i = 0; i < 32; i++) {
			let node = this.tlas[index];
			let left_w = weight_light(node.left, point);
			let right_w = weight_light(node.right, point);
			let total_w = left_w + right_w;
			if (total_w < 0.0000001f)
				return none;
			let left_p = left_w / total_w;
			let right_p = 1.f - left_p;
			if (p < left_p) {
				if (node.L == ~0)
					return none;

				pdf *= left_p;
				p /= left_p;
				index = node.L >> 1;
				if ((node.L & 1) == 0)
					break;
			} else {
				if (node.R == ~0)
					return none;

				pdf *= right_p;
				p = (p - left_p) / right_p;
				index = node.R >> 1;
				if ((node.R & 1) == 0)
					break;
			}
		}

		let instance = this.blases[index];
		index = 0;
		var mesh_point = point;
		let inv_mat = instance.transform.inv_mat();
		mesh_point.position = mul(inv_mat, f32x4(point.position, 1.f)).xyz;
		mesh_point.normal = mul(inv_mat, f32x4(point.normal, 0.f)).xyz;
		for (u32 i = 0; i < 32; i++) {
			let node = instance.blas->root[index];
			let left_w = weight_light(node.left, mesh_point);
			let right_w = weight_light(node.right, mesh_point);
			let total_w = left_w + right_w;
			if (total_w < 0.0000001f)
				return none;
			let left_p = left_w / total_w;
			let right_p = 1.f - left_p;
			if (p < left_p) {
				pdf *= left_p;
				p /= left_p;
				index = node.L >> 1;
				if ((node.L & 1) == 0)
					return this.sample_tri(rng, instance, index, pdf, point);
			} else {
				pdf *= right_p;
				p = (p - left_p) / right_p;
				index = node.R >> 1;
				if ((node.R & 1) == 0)
					return this.sample_tri(rng, instance, index, pdf, point);
			}
		}

		return none;
	}

	// TODO: sample by solid angle?
	LightSample sample_tri(inout Rng rng, BlasInstance instance, u32 index, f32 pdf, LightSamplePoint point) {
		let s = rng.sample2();
		f32x2 b;
		if (s.x < s.y) {
			b.x = s.x / 2.f;
			b.y = s.y - b.x;
		} else {
			b.y = s.y / 2.f;
			b.x = s.x - b.y;
		}

		let bary = f32x3(1.f - (b.x + b.y), b.x, b.y);

		let tri = instance.blas->indices[index];
		let v0 = instance.blas->vertices[tri.x];
		let v1 = instance.blas->vertices[tri.y];
		let v2 = instance.blas->vertices[tri.z];

		let tmat = instance.transform.mat();
		let v0p = mul(tmat, f32x4(v0.position, 1.f)).xyz;
		let v1p = mul(tmat, f32x4(v1.position, 1.f)).xyz;
		let v2p = mul(tmat, f32x4(v2.position, 1.f)).xyz;

		let position = bary.x * v0p + bary.y * v1p + bary.z * v2p;
		let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
		let obj_norm = f32x4(bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal, 0.f);
		var g_normal = cross(v1p - v0p, v2p - v0p);
		let area = length(g_normal) * 0.5f;
		g_normal = normalize(g_normal);

		let mat = instance.blas->material;
		let em = mat->emissive.get();
		let emf = mat->emissive_factor;
		let L = rec709_to_rec2020(em.sample_mip(point.sampler, uv, 0.0, f32x4(1.f)).xyz * emf);

		let r = position - point.position;
		let t2 = dot(r, r);
		let t = sqrt(t2);
		let wi = r / t;
		let d = abs(dot(g_normal, -wi));
		pdf /= area;

		return LightSample(L * d / t2, wi, t, pdf);
	}
}

public struct LightTreeNode {
	public SgLight left;
	public SgLight right;
	// LSB is if is inner node.
	public u32 L;
	// LSB is if is inner node.
	public u32 R;

	public __init(NodeData left, NodeData right, u32 L, u32 R) {
		this.left = SgLight(left);
		this.right = SgLight(right);
		this.L = L;
		this.R = R;
	}
}

public struct SgLight {
	public f32x3 position;
	public f32 variance;
	public f32x3 intensity;
	public f32x3 axis;

	public __init(NodeData data) {
		this.position = data.position.xyz;
		this.variance = data.position.w;
		let sg = average_direction_to_sg_axis(data.normal_average);
		this.intensity = power_to_intensity(data.power, sg.w);
		this.axis = sg.xyz * sg.w;
	}
}

public struct BlasInstance {
	public Transform transform;
	public LightBlas* blas;
}

public struct LightBlas {
	public LightTreeNode* root;
	public Vertex* vertices;
	public u32x3* indices;
	public Material<NonUniform>* material;
	public NodeData data;
}

static const f32 SG_LIGHT_MAX_SHARPNESS = 0x1.0p41;

f32 weight_light(SgLight light, LightSamplePoint point) {
	let light_sharpness = length(light.axis);
	let axis = light_sharpness > 0.00001f ? light.axis / light_sharpness : f32x3(0.f);

	let dir = light.position - point.position;
	let t2 = dot(dir, dir);
	let wi = dir / sqrt(t2);
	let emissive = light.intensity / light.variance;
	let sharpness = light_sharpness / light.variance;
	let lobe = SGProduct(axis, sharpness, wi, sharpness);

	let amplitude = exp(lobe.logAmplitude);
	let cosine = dot(lobe.axis, point.normal);
	let diffuse_illum = amplitude * SGClampedCosineProductIntegralOverPi2024(cosine, lobe.sharpness);
	let color = rec709_to_rec2020(emissive) * point.base_color;
	return diffuse_illum * luminance_rec2020(color);
}

struct SGLobe {
	f32x3 axis;
	f32 sharpness;
	f32 logAmplitude;
}

// https://github.com/yusuketokuyoshi/VSGL/blob/master/VSGL/Shaders/SphericalGaussian.hlsli
// Product of two SGs.
SGLobe SGProduct(const float3 axis1, const float sharpness1, const float3 axis2, const float sharpness2) {
	const float3 axis = axis1 * sharpness1 + axis2 * sharpness2;
	const float sharpness = length(axis);

	// Compute logAmplitude = sharpness - (sharpness1 + sharpness2).
	// Since sharpness - sharpness1 - sharpness2 in floating point arithmetic can produce a significant numerical error,
	// we use a numerically stable form derived by logAmplitude = sharpness - (sharpness1 + sharpness2)
	//              = (||axis1 * sharpness1 + axis2 * sharpness2||^2 - (sharpness1 + sharpness2)^2) / (sharpness +
	//              sharpness1 + sharpness2) = (sharpness1^2 + 2 * sharpness1 * sharpness2 * dot(axis1, axis2) +
	//              sharpness2^2 - (sharpness1^2 + 2 * sharpness1 * sharpness2 + sharpness2^2) / (sharpness + sharpness1
	//              + sharpness2) = 2 * sharpness1 * sharpness2 * (dot(axis1, axis2) - 1) / (sharpness + sharpness1 +
	//              sharpness2) = -sharpness1 * sharpness2 * ||axis1 - axis2||^2 / (sharpness + sharpness1 +
	//              sharpness2).
	const float3 d = axis1 - axis2;
	const float len2 = dot(d, d);  // -0.5 * len2 = dot(axis1, axis2) - 1. Using len2 improves the numerical stability
								   // when axis1 \approx axis2.
	const float logAmplitude = -sharpness1 * sharpness2 * len2 / (sharpness + sharpness1 + sharpness2);

	const SGLobe result = { axis / sharpness, sharpness, logAmplitude };
	return result;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 5]
float UpperSGClampedCosineIntegralOverTwoPi(const float sharpness) {
	if (sharpness <= 0.5) {
		// Taylor-series approximation for the numerical stability.
		// TODO: Derive a faster polynomial approximation.
		return (((((((-1.0 / 362880.0) * sharpness + 1.0 / 40320.0) * sharpness - 1.0 / 5040.0) * sharpness +
				   1.0 / 720.0) *
					  sharpness -
				  1.0 / 120.0) *
					 sharpness +
				 1.0 / 24.0) *
					sharpness -
				1.0 / 6.0) *
				   sharpness +
			   0.5;
	}

	return (expm1(-sharpness) + sharpness) / (sharpness * sharpness);
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 6]
float LowerSGClampedCosineIntegralOverTwoPi(const float sharpness) {
	const float e = exp(-sharpness);

	if (sharpness <= 0.5) {
		// Taylor-series approximation for the numerical stability.
		// TODO: Derive a faster polynomial approximation.
		return e * (((((((((1.0 / 403200.0) * sharpness - 1.0 / 45360.0) * sharpness + 1.0 / 5760.0) * sharpness -
						 1.0 / 840.0) *
							sharpness +
						1.0 / 144.0) *
						   sharpness -
					   1.0 / 30.0) *
						  sharpness +
					  1.0 / 8.0) *
						 sharpness -
					 1.0 / 3.0) *
						sharpness +
					0.5);
	}

	return e * (-expm1(-sharpness) - sharpness * e) / (sharpness * sharpness);
}

// Approximate product integral of an SG and clamped cosine / pi.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 7]
float SGClampedCosineProductIntegralOverPi2024(const float cosine, const float sharpness) {
	// Fitted approximation for t(sharpness).
	const float A = 2.7360831611272558028247203765204;
	const float B = 17.02129778174187535455530451145;
	const float C = 4.0100826728510421403939290030394;
	const float D = 15.219156263147210594866010069381;
	const float E = 76.087896272360737270901154261082;
	const float t =
		sharpness * sqrt(0.5 * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
	const float tz = t * cosine;

	// In this HLSL implementation, we roughly implement erfc(x) = 1 - erf(x) which can have a numerical error for large
	// x. Therefore, unlike the original impelemntation [Tokuyoshi et al. 2024], we clamp the lerp factor with the
	// machine epsilon / 2 for a conservative approximation. This clamping is unnecessary for languages that have a
	// precise erfc function (e.g., C++). The original implementation [Tokuyoshi et al. 2024] uses a precise erfc
	// function and does not clamp the lerp factor.
	const float INV_SQRTPI = 0.56418958354775628694807945156077;  // = 1/sqrt(pi).
	const float CLAMPING_THRESHOLD = 0.5 * 1e-5;				  // Set zero if a precise erfc function is available.
	const float lerpFactor =
		saturate(max(0.5 * (cosine * erfc(-tz) + erfc(t)) -
						 0.5 * INV_SQRTPI * exp(-tz * tz) * expm1(t * t * (cosine * cosine - 1.0)) / t,
					 CLAMPING_THRESHOLD));

	// Interpolation between lower and upper hemispherical integrals.
	const float lowerIntegral = LowerSGClampedCosineIntegralOverTwoPi(sharpness);
	const float upperIntegral = UpperSGClampedCosineIntegralOverTwoPi(sharpness);
	return 2.0 * lerp(lowerIntegral, upperIntegral, lerpFactor);
}

float mulsign(const float x, const float y) {
	return asfloat((asuint(y) & 0x80000000) ^ asuint(x));
}

float2 mulsign(const float2 x, const float2 y) {
	return asfloat((asuint(y) & 0x80000000) ^ asuint(x));
}

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

// exp(x) - 1 with cancellation of rounding errors.
// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p.19]
float expm1(const float x) {
	const float u = exp(x);

	if (u == 1.0) {
		return x;
	}

	const float y = u - 1.0;

	if (abs(x) < 1.0) {
		return y * x / log(u);
	}

	return y;
}

float erf(const float x) {
	// Early return for large |x|.
	if (abs(x) >= 4.0) {
		return mulsign(1.0, x);
	}

	// Polynomial approximation based on the approximation posted in
	// https://forums.developer.nvidia.com/t/optimized-version-of-single-precision-error-function-erff/40977
	if (abs(x) > 1.0) {
		// The maximum error is smaller than the approximation described in Abramowitz and Stegun [1964 "Handbook of
		// Mathematical Functions with Formulas, Graphs, and Mathematical Tables", 7.1.26, p.299].
		const float A1 = 1.628459513;
		const float A2 = 9.15674746e-1;
		const float A3 = 1.54329389e-1;
		const float A4 = -3.51759829e-2;
		const float A5 = 5.66795561e-3;
		const float A6 = -5.64874616e-4;
		const float A7 = 2.58907676e-5;
		const float a = abs(x);
		const float y = 1.0 - exp2(-(((((((A7 * a + A6) * a + A5) * a + A4) * a + A3) * a + A2) * a + A1) * a));

		return mulsign(y, x);
	}

	// The maximum error is smaller than the 6th order Taylor polynomial.
	const float A1 = 1.128379121;
	const float A2 = -3.76123011e-1;
	const float A3 = 1.12799220e-1;
	const float A4 = -2.67030653e-2;
	const float A5 = 4.90735564e-3;
	const float A6 = -5.58853149e-4;
	const float x2 = x * x;

	return (((((A6 * x2 + A5) * x2 + A4) * x2 + A3) * x2 + A2) * x2 + A1) * x;
}

// Complementary error function erfc(x) = 1 - erf(x).
// This implementation can have a numerical error for large x.
// TODO: Precise implementation.
float erfc(const float x) {
	return 1.0 - erf(x);
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

	public __init(f32x3 vertices[3], Material<NonUniform>* material) {
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
		let w_left = luminance_rec709(left.power / (left.power + right.power));
		let w_right = 1.f - w_left;
		this.aabb_min = min(left.aabb_min, right.aabb_min);
		this.aabb_max = max(left.aabb_max, right.aabb_max);
		this.position.xyz = left.position.xyz * w_left + right.position.xyz * w_right;

		let r = length(this.aabb_max - this.aabb_min) * 0.5f;
		// let pos_diff = left.position.xyz - right.position.xyz;
		// this.position.w += w_left * w_right * dot(pos_diff, pos_diff);
		this.position.w = r * r * 0.5f;

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
