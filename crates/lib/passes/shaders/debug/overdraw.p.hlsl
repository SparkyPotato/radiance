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
    f32 scaled = heat * 6.f;
    u32 bot = u32(floor(scaled));
    u32 top = u32(ceil(scaled));
    return lerp(cols[bot], cols[top], scaled - f32(bot));
}

float4 main(VertexOutput input): SV_Target0 {
	uint2 pixel = Constants.overdraw.pixel_of_uv(input.uv);
	u32 value = asuint(Constants.overdraw.load(pixel).x);
	f32 heat = clamp((f32(value) - f32(Constants.bottom)) / f32(Constants.top - Constants.bottom), 0.f, 1.f);
	return float4(heatmap(heat), 1.f);
}