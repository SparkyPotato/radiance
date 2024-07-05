#include "radiance-graph/types.l.hlsl"

#include "screen.l.hlsl"

VertexOutput main(u32 vertex: SV_VertexID) {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    float4 position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);

    VertexOutput ret = { position, uv };
    return ret;
}
