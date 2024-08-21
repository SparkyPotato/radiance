#include "radiance-graph/types.l.hlsl"

[[vk::binding(2, 0)]] RWTexture2D<u64> Textures[];

struct PushConstants {
	[[vk::offset(16)]] u32 output;
};

PUSH PushConstants Constants;

void main(u32 data: VisBuffer, float4 pos: SV_Position) {
	f32 depth = pos.z * pos.w;
	uint2 out_pos = uint2(pos.xy);
	u64 encoded = (u64(asuint(depth)) << 32) | u64(data);
	InterlockedMax(Textures[Constants.output][out_pos], encoded);
}
