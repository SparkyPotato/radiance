#include "radiance-graph/types.l.hlsl"

[[vk::binding(2, 0)]] RWTexture2D<u64> Textures[];
[[vk::binding(2, 0)]] RWTexture2D<u32> OTextures[];

struct PushConstants {
	[[vk::offset(16)]] u32 output;
	[[vk::offset(20)]] u32 overdraw;
};

PUSH PushConstants Constants;

void main(u32 data: VisBuffer, float4 pos: SV_Position) {
	f32 depth = pos.z * pos.w;
	uint2 out_pos = uint2(pos.xy);
	u64 encoded = (u64(asuint(depth)) << 32) | u64(data);
	InterlockedAdd(OTextures[Constants.overdraw][out_pos], 1);
	InterlockedMax(Textures[Constants.output][out_pos], encoded);
}