#include "common.l.hlsl"

struct VertexInput {
    float2 position;
    float2 uv;
    uint color;
};

float4 unpack_color(uint color) {
    return float4(
        float(color & 255),
        float((color >> 8) & 255),
        float((color >> 16) & 255),
        float((color >> 24) & 255)
    ) / 255.f;
}

float4 project_screenspace(float2 screenspace) {
    return float4(
        2.f * screenspace / float2(Constants.screen_size) - 1.f,
        0.f,
        1.f
    );
}

VertexOutput main(uint vertex: SV_VertexID) {
    VertexInput input = Buffers[Constants.vertex_buffer_id].Load<VertexInput>(vertex * sizeof(VertexInput));

    VertexOutput output;
    output.position = project_screenspace(input.position);
    output.uv = input.uv;
    output.color = unpack_color(input.color);

    return output;
}
