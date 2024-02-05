#include "common.l.hlsl"

struct PushConstants {
    [[vk::offset(12)]] Tex2D image;
    Sampler sampler;
};

PUSH PushConstants Constants;

float4 main(VertexOutput input): SV_Target {
    return Constants.image.sample(Constants.sampler, input.uv) * input.color;
}
