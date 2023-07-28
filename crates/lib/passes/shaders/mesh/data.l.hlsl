#pragma once

#include "radiance-core/interface.l.hlsl"

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
};

struct Cone {
    u32 apex;
    u32 axis_cutoff;
};

struct Meshlet {
	f32 aabb_min[3];
	f32 aabb_extent[3];
	Cone cone;
    u32 vertex_byte_offset;
    u32 index_byte_offset;
    u16 vert_and_tri_count;
    u16 _pad[3];
};

struct Camera {
    float4x4 view;
    float4x4 proj;
};
