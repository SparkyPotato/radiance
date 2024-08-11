#pragma once

#include "radiance-passes/asset/data.l.hlsl"

struct PushConstants {
    Buf<Instance> instances;
    Buf<MeshletPointer> meshlet_pointers;
    Buf<Camera> camera;
    Sampler hzb_sampler;
    Tex2D hzb;
    Buf<u32> culled;
    u32 meshlet_count;
    u32 width;
    u32 height;
};

struct MeshPayload {
    u32 base;
    u32 pointers[16];

    u32 get(u32 index) {
        u32 i = index >> 2;
        u32 o = (index & 0b11) << 3;
        return this.base + ((this.pointers[i] >> o) & 0xff);
    }
};

struct VertexOutput {
    float4 position: SV_Position;
};

struct PrimitiveOutput {
    [[vk::location(0)]] u32 data: VisBuffer;
};
