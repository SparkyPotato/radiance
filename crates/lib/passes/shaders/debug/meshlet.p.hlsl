#include "radiance-graph/interface.l.hlsl"
#include "radiance-graph/util/screen.l.hlsl"
#include "radiance-passes/mesh/visbuffer/visbuffer.l.hlsl"

struct PushConstants {
    Tex2D visbuffer;
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

float4 main(VertexOutput input): SV_Target0 {
    uint2 pixel = Constants.visbuffer.pixel_of_uv(input.uv);
    u32 value = asuint(Constants.visbuffer.load(pixel).x);
    if (value == 0xffffffff) discard;

    VisBufferData data = VisBufferData::decode(value);
    u32 h = hash(value);
    float3 color = float3(float(h & 255), float((h >> 8) & 255), float((h >> 16) & 255));
    return float4(color / 255.0, 1.0);
}
