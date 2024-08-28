#include "visbuffer.l.hlsl"

struct PushConstants {
	[[vk::offset(16)]] VisBufferTex output;
};

PUSH PushConstants Constants;

void main(u32 data: VisBuffer, float4 pos: SV_Position) {
	f32 depth = pos.z * pos.w;
	uint2 out_pos = uint2(pos.xy);
	Constants.output.write(out_pos, depth, data, false);
}
