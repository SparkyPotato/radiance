#include "common.l.hlsl"

float3 heatmap(f32 heat) {
	float3 cols[7] = {
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 1.f),
		float3(0.f, 1.f, 1.f),
		float3(0.f, 1.f, 0.f),
		float3(1.f, 1.f, 0.f),
		float3(1.f, 0.f, 0.f),
		float3(1.f, 1.f, 1.f),
	};
	for (int i = 1; i < 7; i++) {
		if (heat < i / 6.f) return lerp(cols[i - 1], cols[i], (heat - (i - 1) / 6.f) * 6.f);
	}
	return cols[6];
}

float4 main(VertexOutput input): SV_Target0 {
	uint2 pixel = Constants.overdraw.pixel_of_uv(input.uv);
	u32 value = asuint(Constants.overdraw.load(pixel).x);
	f32 heat = max((f32(value) - f32(Constants.bottom)) / f32(Constants.top - Constants.bottom), 0.f);
	return float4(heatmap(heat), 1.f);
}