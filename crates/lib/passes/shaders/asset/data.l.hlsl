#pragma once

#include "radiance-graph/interface.l.hlsl"

struct Vertex {
    f32 position[3];
    f32 normal[3];
    f32 tangent[4];
    f32 uv[2];
};

struct MeshletPointer {
    u32 instance;
    u32 meshlet;
};

struct Pos {
    f32 pos[3];
};

struct Instance {
    f32 transform[12];
    Buf<bytes> mesh;

    float4x4 get_transform() {
        f32 t[12] = this.transform;
        float4x4 ret = {
            t[0], t[3], t[6], t[9],
            t[1], t[4], t[7], t[10],
            t[2], t[5], t[8], t[11],
            0.f,  0.f,  0.f,  1.f,
        };
        return ret;
    }
};

struct Meshlet {
    u32 vertex_offset;
    u32 index_offset;
    u16 vert_and_tri_count;
    u16 _pad;
    f32 bounding[4];
    f32 group_error[4];
    f32 parent_error[4];
};

struct Material {
    f32 base_color_factor[4];
    f32 metallic_factor;
    f32 roughness_factor;
    f32 emissive_factor[3];
};

#define CULL_CAMERA 0
#define DRAW_CAMERA 1

struct Camera {
    float4x4 view;
    float4x4 proj;
    float4x4 view_proj;
    f32 cot_fov;
    f32 _pad[15];
};
