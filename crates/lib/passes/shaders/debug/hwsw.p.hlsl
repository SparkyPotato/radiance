#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target0 {
	u32 data = Constants.read.hwsw(input.uv);

	float3 cols[3] = {
		float3(0.f, 0.f, 0.f),
		float3(0.f, 1.f, 0.f),
		float3(1.f, 0.f, 0.f)
	};
	if (data > 2) return float4(1.f, 0.f, 1.f, 1.f);
	return float4(cols[data], 1.f);
}