#pragma once

#include "radiance-graph/interface.l.hlsl"

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct MeshletPointer {
    u32 instance;
    u32 meshlet;
    u32 meshlet_count;

    static MeshletPointer get(Buf<bytes> instances, u32 instance_count, u32 id) {
        u32 left = 0;
        u32 right = instance_count;
        while (left < right) {
            u32 mid = (left + right) >> 1;
            if (id >= instances.load<u32>(0, mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        MeshletPointer ret;
        ret.instance = right;
        u32 prev = right != 0 ? instances.load<u32>(0, right - 1) : 0;
        ret.meshlet = id - prev;
        ret.meshlet_count = instances.load<u32>(0, right) - prev;
        return ret;
    }
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

    static Instance get(Buf<bytes> instances, u32 instance_count, u32 id) {
        return instances.load<Instance>(sizeof(u32) * instance_count, id);
    }
};

struct MeshletBounds {
    float4 bounding;
    float4 group_error;
    float4 parent_error;

    static MeshletBounds get(Buf<bytes> mesh, u32 id) {
        return mesh.load<MeshletBounds>(0, id);
    }
};

struct MeshletData {
    u32 vertex_offset;
    u32 index_offset;
    u16 vert_and_tri_count;
    u16 _pad;

    static MeshletData get(Buf<bytes> mesh, u32 meshlet_count, u32 id) {
        return mesh.load<MeshletData>(sizeof(MeshletBounds) * meshlet_count, id);
    }

    u32 vertex_count() {
        return this.vert_and_tri_count & 0xff;
    }

    u32 tri_count() {
        return (this.vert_and_tri_count >> 8) & 0xff;
    }

    Vertex vertex(Buf<bytes> mesh, u32 vertex) {
        return mesh.load<Vertex>(this.vertex_offset, vertex);
    }

    u32 tri(Buf<bytes> mesh, u32 tri) {
        return mesh.load<u32>(this.index_offset, tri);
    }
};

struct Material {
    float4 base_color_factor;
    f32 metallic_factor;
    f32 roughness_factor;
    float3 emissive_factor;
};

struct Camera {
    float4x4 view;
    float4x4 view_proj;
    f32 w;
    f32 h;
    f32 near;
    f32 _pad;
    float4 frustum;
};
