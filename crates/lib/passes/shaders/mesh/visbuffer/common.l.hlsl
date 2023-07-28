#pragma once

#include "radiance-passes/mesh/data.l.hlsl"

struct PushConstants {
    Buf<Instance> instances;
    Buf<MeshletPointer> meshlet_pointers;
    Buf<Camera> camera;
    u32 meshlet_count;
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
