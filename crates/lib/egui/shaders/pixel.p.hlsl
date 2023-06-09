#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target {
    return Constants.image.sample(Constants.sampler, input.uv) * input.color;
}
