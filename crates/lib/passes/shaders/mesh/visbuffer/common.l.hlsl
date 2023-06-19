#include "radiance-core/types.l.hlsl"

struct VertexOutput {
    float4 position: SV_Position;
    u32 meshlet_pointer_id: ID;
};
