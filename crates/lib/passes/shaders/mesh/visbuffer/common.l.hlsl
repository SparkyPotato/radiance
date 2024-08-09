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

struct PointerWithId {
    MeshletPointer pointer;
    u32 id;
};

struct MeshPayload {
    PointerWithId pointers[64];
};

struct VertexOutput {
    float4 position: SV_Position;
};

struct PrimitiveOutput {
    [[vk::location(0)]] u32 data: VisBuffer;
};
