#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target {
    return Textures[Constants.image_id].Sample(Samplers[Constants.sampler_id], input.uv) * input.color;
}
