#pragma once

#include "radiance-passes/mesh/data.l.hlsl"

struct PushConstants {
    Buf<Instance> instances;
    Buf<MeshletPointer> meshlet_pointers;
    Buf<u32> rw;
    Buf<u32> ww;
    Buf<u32> rd;
    Buf<u32> wd;
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

#define VISIBLE_WORKGROUP_OFFSET 0
#define VISIBLE_DATA_OFFSET 0
#define INVISIBLE_WORKGROUP_OFFSET 4
#define INVISIBLE_DATA_OFFSET Constants.meshlet_count
