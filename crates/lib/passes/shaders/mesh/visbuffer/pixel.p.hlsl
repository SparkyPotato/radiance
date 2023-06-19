#include "common.l.hlsl"
#include "visbuffer.l.hlsl"

u32 main(VertexOutput input, u32 triangle_id: SV_PrimitiveID): SV_Target0 {
    VisBufferData data = { input.meshlet_pointer_id, triangle_id };
    return data.encode();
}
