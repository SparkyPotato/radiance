#include "radiance-graph/interface.l.hlsl"
#include "radiance-graph/util/screen.l.hlsl"
#include "radiance-passes/mesh/visbuffer.l.hlsl"
#include "radiance-passes/mesh/cull.l.hlsl"

struct PushConstants {
	u32 visbuffer;
	Tex2D overdraw;
	MeshletQueue early;
	MeshletQueue late;
	u32 bottom;
	u32 top;
};

[[vk::binding(1, 0)]] Texture2D<u64> Inputs[];

PUSH PushConstants Constants;

// Stolen from https://gist.github.com/badboy/6267743 and niagara.
// Thanks copilot.
u32 hash(u32 a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

VisBufferData load_visbuffer(float2 uv) {
	u32 width, height;
	Inputs[Constants.visbuffer].GetDimensions(width, height);
	float2 dim = float2(width, height);
	float x = round(uv.x * dim.x - 0.5f);
	float y = round(uv.y * dim.y - 0.5f);
	uint2 pos = uint2(x, y);
	u64 data = Inputs[Constants.visbuffer].Load(uint3(pos, 0));
	return VisBufferData::decode(u32(data));
}
