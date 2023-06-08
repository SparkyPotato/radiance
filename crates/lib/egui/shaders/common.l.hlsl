struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: UV;
    float4 color: COLOR;
};

struct PushConstants {
    uint2 screen_size;
    uint vertex_buffer_id;
    uint image_id;
    uint sampler_id;
};

[[vk::push_constant]] PushConstants Constants;
[[vk::binding(0, 0)]] ByteAddressBuffer Buffers[];
[[vk::binding(1, 0)]] Texture2D Textures[];
[[vk::binding(3, 0)]] SamplerState Samplers[];
