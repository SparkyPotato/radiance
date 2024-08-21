#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target0 {
	uint2 pixel = Constants.visbuffer.pixel_of_uv(input.uv);
	u32 value = asuint(Constants.overdraw.load(pixel).x);
	f32 heat = f32(value - Constants.bottom) / f32(Constants.top - Constants.bottom);
	return float4(heat, heat, heat, 1.0);
}