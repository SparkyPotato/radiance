import graph;
import graph.util.color;
import graph.util.rng;

struct SgLight {
	f32x3 pos;
	f32 variance;
	f32x3 intensity;
	f32x3 axis;
	f32 sharpness;
}

public struct LightTreeNode {
	SgLight left;
	SgLight right;
	// If indices are u32::MAX, then the relevant node is a point light.
	// If indices have their MSB set, then the node is an emissive.
	u32 left_index;
	u32 right_index;
}

public struct LightSample {
	public f32x3 L;
	public f32x3 wi_world;
	public f32 t;
	public f32 pdf;
	public bool punctual;
}

public interface SamplerInfo {
	f32x3 base_color();
	f32x3 pos();
	f32x3 normal();
}

public interface SamplerInterface : SamplerInfo {
	LightSample sample_emissive(inout Rng rng, u32 index);
}

LightSample sample_point_light(SgLight light, f32x3 pos) {
	let dir = light.pos - pos;
	let t2 = dot(dir, dir);
	let L = rec709_to_rec2020(light.intensity * 2.0 * PI / t2);
	let t = sqrt(t2);
	let wi = dir / t;
	return { L, wi, t, 1.f, true };
}

// https://github.com/yusuketokuyoshi/VSGL/blob/master/VSGL/Shaders/LightingPS.hlsl
f32 weight_light(SgLight light, SamplerInfo s) {
	let dir = light.pos - s.pos();
	let t2 = dot(dir, dir);
	let wi = dir / sqrt(t2);
	let emissive = light.intensity / light.variance;
	let sharpness = light.sharpness / light.variance;
	let lobe = SGProduct(light.axis, light.sharpness, wi, sharpness);

	let amplitude = exp(lobe.logAmplitude);
	let cosine = dot(lobe.axis, s.normal());
	let diffuse_illum = amplitude * SGClampedCosineProductIntegralOverPi2024(cosine, lobe.sharpness);
	let color = rec709_to_rec2020(emissive) * s.base_color();
	return diffuse_illum * luminance_rec2020(color);
}

interface TraversalCallback<T> {
	bool should_go_left(f32 left_p);
	[mutating]
	void next_left(f32 left_p, f32 right_p);
	[mutating]
	void next_right(f32 left_p, f32 right_p);

	T point_light(SgLight light, f32 pdf);
	[mutating]
	T emissive(u32 index, f32 pdf);
	T default();
}

struct SampleCallback : TraversalCallback<LightSample> {
	Rng rng;
	f32 p;
	SamplerInterface s;

	bool should_go_left(f32 left_p) {
		return p < left_p;
	}
	[mutating]
	void next_left(f32 left_p, f32 right_p) {
		this.p /= left_p;
	}
	[mutating]
	void next_right(f32 left_p, f32 right_p) {
		this.p = (this.p - left_p) / right_p;
	}

	LightSample point_light(SgLight light, f32 pdf) {
		var ls = sample_point_light(light, s.pos());
		ls.pdf *= pdf;
		return ls;
	}
	[mutating]
	LightSample emissive(u32 index, f32 pdf) {
		var ls = s.sample_emissive(this.rng, index);
		ls.pdf *= pdf;
		return ls;
	}
	LightSample default() {
		return {};
	}
}

struct PdfCallback : TraversalCallback<f32> {
	u32 path;

	bool should_go_left(f32 left_p) {
		return (path & 1) == 0;
	}
	[mutating]
	void next_left(f32 left_p, f32 right_p) {
		this.path >>= 1;
	}
	[mutating]
	void next_right(f32 left_p, f32 right_p) {
		this.path >>= 1;
	}

	f32 point_light(SgLight light, f32 pdf) {
		return 0.f;
	}
	f32 emissive(u32 index, f32 pdf) {
		return pdf;
	}
	f32 default() {
		return 0.f;
	}
}

public struct LightTree {
	LightTreeNode* root;
	u32* emissive_paths;

	R traverse<T, R>(SamplerInfo s, inout T cb)
		where T : TraversalCallback<R> {
		var node = this.root[0];
		var pdf = 1.f;
		for (int i = 0; i < 32; i++) {
			let left_w = weight_light(node.left, s);
			let right_w = weight_light(node.right, s);
			let left_p = left_w / (left_w + right_w);
			let right_p = 1.f - left_p;
			if (cb.should_go_left(left_p)) {
				pdf *= left_p;
				if (node.left_index == ~0) {
					return cb.point_light(node.left, pdf);
				} else if ((node.left_index >> 31) == 1) {
					return cb.emissive(node.left_index & ~(1 << 31), pdf);
				}
				node = this.root[node.left_index];
				cb.next_left(left_p, right_p);
			} else {
				pdf *= right_p;
				if (node.right_index == ~0) {
					return cb.point_light(node.right, pdf);
				} else if ((node.right_index >> 31) == 1) {
					return cb.emissive(node.right_index & ~(1 << 31), pdf);
				}
				node = this.root[node.right_index];
				cb.next_right(left_p, right_p);
			}
		}
		return cb.default();
	}

	public LightSample sample(inout Rng rng, SamplerInterface s) {
		let p = rng.sample();
		SampleCallback cb = { rng, p, s };
		let ret = this.traverse(s, cb);
		rng = cb.rng;
		return ret;
	}

	public f32 pdf_of_emissive(u32 index, SamplerInfo s) {
		PdfCallback cb = { this.emissive_paths[index] };
		return this.traverse(s, cb);
	}
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
