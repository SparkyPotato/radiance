#pragma once

#include "radiance-core/interface.l.hlsl"

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: UV;
};
