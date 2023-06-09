#include "radiance-core/interface.l.hlsl"

struct VertexInput {
    float2 position;
    float2 uv;
    u32 color;
};

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: UV;
    float4 color: COLOR;
};

struct PushConstants {
    uint2 screen_size;
    Buf<VertexInput> vertex_buffer;
    Tex2D image;
    Sampler sampler;
};

PUSH PushConstants Constants;
