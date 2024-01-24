#include "common.l.hlsl"

struct PushConstants {
    float2 screen_size;
    Buf<VertexInput> vertices;
};

PUSH PushConstants Constants;

float4 unpack_color(u32 color) {
    return float4(
        float(color & 255),
        float((color >> 8) & 255),
        float((color >> 16) & 255),
        float((color >> 24) & 255)
    ) / 255.f;
}

float4 project_screenspace(float2 screenspace) {
    return float4(
        2.f * screenspace / Constants.screen_size - 1.f,
        0.f,
        1.f
    );
}

VertexOutput main(u32 vertex: SV_VertexID) {
    VertexInput input = Constants.vertices.load(vertex);

    VertexOutput output;
    output.position = project_screenspace(input.position);
    output.uv = input.uv;
    output.color = unpack_color(input.color);

    return output;
}
