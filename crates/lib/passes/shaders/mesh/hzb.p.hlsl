#include "radiance-core/interface.l.hlsl"

struct PushConstants {
    Tex2D depth;
    u32 prev_level;
    uint2 prev_size;
};

PUSH PushConstants Constants;

f32 main(float4 coord: SV_Position): SV_Depth {
    uint2 curr_coord = uint2(coord.xy);
    uint2 prev_coord = curr_coord * 2;
    float4 values;
    values.x = Constants.depth.load(prev_coord).x;
    values.y = Constants.depth.load(prev_coord + uint2(1, 0)).x;
    values.z = Constants.depth.load(prev_coord + uint2(1, 1)).x;
    values.w = Constants.depth.load(prev_coord + uint2(0, 1)).x;
    float min_depth = min(min(values.x, values.y), min(values.z, values.w));

    bool should_include_col = (Constants.prev_size.x & 1) != 0;
    bool should_include_row = (Constants.prev_size.y & 1) != 0;
    if (should_include_col) {
        float2 extra;
        extra.x = Constants.depth.load(curr_coord + uint2(2, 0)).x;
        extra.y = Constants.depth.load(curr_coord + uint2(2, 1)).x;
        if (should_include_row) {
            float corner = Constants.depth.load(curr_coord + uint2(2, 2)).x;
            min_depth = min(min_depth, corner);
        }
        min_depth = min(min_depth, min(extra.x, extra.y));
    }
    if (should_include_row) {
        float2 extra;
        extra.x = Constants.depth.load(curr_coord + uint2(0, 2)).x;
        extra.y = Constants.depth.load(curr_coord + uint2(1, 2)).x;
        min_depth = min(min_depth, min(extra.x, extra.y));
    }

    return min_depth;
}
