#pragma once

#include "radiance-passes/asset/data.l.hlsl"
#include "radiance-passes/mesh/cull.l.hlsl"

struct PushConstants {
    Buf<Instance> instances;
    Buf<MeshletPointer> meshlet_pointers;
    Buf<Camera> camera;
    u32 meshlet_count;
    u32 resolution;
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
