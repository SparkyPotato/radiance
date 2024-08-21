#include "radiance-graph/interface.l.hlsl"
#include "radiance-graph/util/screen.l.hlsl"
#include "radiance-passes/mesh/visbuffer.l.hlsl"
#include "radiance-passes/mesh/cull.l.hlsl"

struct PushConstants {
	Tex2D visbuffer;
	MeshletQueue early;
	MeshletQueue late;
};

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
