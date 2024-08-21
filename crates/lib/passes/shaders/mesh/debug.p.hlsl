#include "radiance-graph/types.l.hlsl"

[[vk::binding(2, 0)]] RWTexture2D<u32> Textures[];

struct PushConstants {
	[[vk::offset(16)]] u32 overdraw;
};

PUSH PushConstants Constants;

u32 main(u32 data: VisBuffer, float4 pos: SV_Position): SV_Target0 {
	InterlockedAdd(Textures[Constants.overdraw][uint2(pos.xy)], 1);
	return data;
}