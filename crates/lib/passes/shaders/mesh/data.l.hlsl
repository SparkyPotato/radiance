#pragma once

#include "radiance-core/interface.l.hlsl"
#include "cull.l.hlsl"

struct Vertex {
	u16 position[3];
	u16 normal[3];
	u16 uv[2];
};

struct MeshletPointer {
    u32 instance;
    u32 meshlet;
};

struct Instance {
    f32 transform[12];
    Buf<Bytes, NonUniform> mesh;
    u32 _pad[3];

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
	f32 aabb_min[3];
	f32 aabb_extent[3];
	u32 vertex_offset;
	u32 index_offset;
	u16 vert_and_tri_count;
	u16 pad;

    Aabb get_mesh_aabb() {
        Aabb ret;
        ret.min = float4(this.aabb_min, 1.f);
        ret.extent = float4(this.aabb_extent, 0.f);
        ret.max = ret.min + ret.extent;
        return ret;
    }
};

#define CULL_CAMERA 0
#define DRAW_CAMERA 1

struct Camera {
    float4x4 view;
    float4x4 proj;
    float4x4 view_proj;
};
