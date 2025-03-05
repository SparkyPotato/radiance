module agx;

// https://iolite-engine.com/blog_posts/minimal_agx_implementation
// https://github.com/Sunset-Flock/Timberdoodle/blob/main/src/rendering/tasks/write_swapchain.hlsl

import graph;
import graph.util;
import graph.util.color;
import passes.tonemap.common;

static const f32x3x3 AGX_IN_MAT = transpose(float3x3(float3(0.856627153315983, 0.137318972929847, 0.11189821299995),
													 float3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
													 float3(0.0482516061458583, 0.101439036467562, 0.811302368396859)));
// explicit AgXOutsetMatrix generated from Filaments AgXOutsetMatrixInv
static const f32x3x3 AGX_OUT_MAT =
	transpose(float3x3(float3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
					   float3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
					   float3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405)));
static const f32x3x3 LINEAR_REC2020_TO_LINEAR_SRGB = transpose(
	float3x3(float3(1.6605, -0.1246, -0.0182), float3(-0.5876, 1.1329, -0.1006), float3(-0.0728, -0.0083, 1.1187)));
static const f32x3x3 LINEAR_SRGB_TO_LINEAR_REC2020 =
	transpose(float3x3(float3(0.6274, 0.0691, 0.0164), float3(0.3293, 0.9195, 0.0880), float3(0.0433, 0.0113, 0.8956)));

// TODO: Make this work in Rec709 and Rec2020.

struct AgX : IFunc<f32x3, f32x3> {
	f32x3 offset;
	f32x3 slope;
	f32x3 power;
	f32 sat;

	f32x3 agx_contrast_approx(f32x3 x) {
		let x2 = x * x;
		let x4 = x2 * x2;
		return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x -
			   0.00232f;
	}

	f32x3 agx(f32x3 val) {
		let min_ev = -12.47393f;
		let max_ev = 4.026069f;
		val = mul(LINEAR_SRGB_TO_LINEAR_REC2020, val);
		val = mul(AGX_IN_MAT, val);
		val = clamp(log2(val), min_ev, max_ev);
		val = (val - min_ev) / (max_ev - min_ev);
		return agx_contrast_approx(val);
	}

	f32x3 agx_look(f32x3 val) {
		let lum = luminance_rec2020(val);
		val = pow(val * this.slope + this.offset, this.power);
		return lum + this.sat * (val - lum);
	}

	f32x3 agx_eotf(f32x3 val) {
		val = mul(AGX_OUT_MAT, val);
		val = pow(val, 2.2f);
		val = mul(LINEAR_REC2020_TO_LINEAR_SRGB, val);
		return clamp(val, 0.0f, 1.0f);
	}

	f32x3 operator()(f32x3 color) {
		color = agx(color);
		color = agx_look(color);
		return agx_eotf(color);
	}
}

struct PushConstants {
	TonemapInput i;
	AgX agx;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
f32x4 main(ScreenOutput input) : SV_Target0 {
	return tonemap(input, Constants.i, Constants.agx);
}
