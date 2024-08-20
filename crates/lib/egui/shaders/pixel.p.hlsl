#include "common.l.hlsl"

struct PushConstants {
	[[vk::offset(12)]] Tex2D image;
	Sampler sampler;
};

PUSH PushConstants Constants;

float3 srgb_gamma_from_linear(float3 rgb) {
	bool3 cutoff = rgb < 0.0031308;
	float3 lower = rgb * 12.92;
	float3 higher = 1.055 * pow(rgb, 1.0 / 2.4) - 0.055;
	return lerp(higher, lower, cutoff);
}

// 0-1 sRGBA gamma  from  0-1 linear
float4 srgba_gamma_from_linear(float4 rgba) {
	return float4(srgb_gamma_from_linear(rgba.rgb), rgba.a);
}

float4 main(VertexOutput input): SV_Target {
	return Constants.image.sample(Constants.sampler, input.uv) * input.color;
}
