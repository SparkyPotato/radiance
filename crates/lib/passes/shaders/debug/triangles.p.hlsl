#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target0 {
	VisBufferData data = Constants.read.data(input.uv);
	if (data.meshlet_id == 0xffffffff >> 7) discard;
	u32 h = hash(data.triangle_id);
	float3 color = float3(float(h & 255), float((h >> 8) & 255), float((h >> 16) & 255));
	return float4(color / 255.0, 1.0);
}
