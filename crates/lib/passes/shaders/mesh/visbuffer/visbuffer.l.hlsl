#pragma once

struct VisBufferData {
    u32 meshlet_pointer_id;
    u32 triangle_id;

    u32 encode() {
        return ((meshlet_pointer_id + 1) << 7) | triangle_id;
    }

    static VisBufferData decode(u32 data) {
        VisBufferData result;
        result.meshlet_pointer_id = (data >> 7) - 1;
        result.triangle_id = data & 0x7f;
        return result;
    }
};
