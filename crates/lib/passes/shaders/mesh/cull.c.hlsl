#include "radiance-passes/mesh/data.l.hlsl"

struct Command {
    u32 index_count;
    u32 instance_count;
    u32 first_index;
    i32 vertex_offset;
    // Use instance ID to tell the shader which instance of the meshlet it is.
    u32 first_instance;
};

struct PushConstants {
    Buf<Meshlet> meshlets;
    Buf<Instance> instances;
    Buf<MeshletPointer> meshlet_pointers;
    Buf<Command> commands;
    // 0: draw count
    Buf<u32> util;
    Buf<Camera> camera;
    u32 meshlet_count;
};

PUSH PushConstants Constants;

struct Aabb {
    float4 min;
    float4 max;
};

bool frustrum_cull(float4x4 mvp, Aabb aabb) {
    float4 aabb_min_clip = mul(mvp, aabb.min);
    float4 aabb_max_clip = mul(mvp, aabb.max);

    float4 aabb_min_ndc = aabb_min_clip / aabb_min_clip.w;
    float4 aabb_max_ndc = aabb_max_clip / aabb_max_clip.w;

    return (aabb_min_clip.w > 0.f && aabb_max_clip.w > 0.f)
        && (aabb_min_ndc.x <= 1.f && aabb_max_ndc.x >= -1.f)
        && (aabb_min_ndc.y <= 1.f && aabb_max_ndc.y >= -1.f)
        && (aabb_min_ndc.z <= 1.f && aabb_max_ndc.z >= 0.f);
}

bool cone_cull(float4x4 mvp, u32 cone) {
    u16 axis_x = u16((cone >> 0) & 0xff);
    u16 axis_y = u16((cone >> 8) & 0xff);
    u16 axis_z = u16((cone >> 16) & 0xff);
    u16 cutoff = u16((cone >> 24) & 0xff);

    u16 x_sign = axis_x & 0x80;
    u16 y_sign = axis_y & 0x80;
    u16 z_sign = axis_z & 0x80;
    u16 c_sign = cutoff & 0x80;

    u16 signed_x = (x_sign << 8) | axis_x;
    u16 signed_y = (y_sign << 8) | axis_y;
    u16 signed_z = (z_sign << 8) | axis_z;
    u16 signed_cutoff = (c_sign << 8) | cutoff;

    float4 norm_cone = float4(signed_x, signed_y, signed_z, signed_cutoff) / 127.f;
    float3 axis = mul(mvp, float4(norm_cone.xyz, 0.f)).xyz;
    float3 view = float3(0.f, 0.f, 1.f);
    return dot(axis, view) <= norm_cone.w;
}

[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    u32 index = id.x;
    if (index >= Constants.meshlet_count) {
        return;
    }

    MeshletPointer pointer = Constants.meshlet_pointers.load(index);
    Instance instance = Constants.instances.load(pointer.instance);
    u32 meshlet_id = instance.base_meshlet + pointer.meshlet;
    Meshlet meshlet = Constants.meshlets.load(meshlet_id);
    Camera camera = Constants.camera.load(0);

    float t[12] = instance.transform;
    float4x4 transform = {
        t[0], t[3], t[6], t[9],
        t[1], t[4], t[7], t[10],
        t[2], t[5], t[8], t[11],
        0.f, 0.f, 0.f, 1.f,
    };
    float4x4 mvp = mul(camera.view_proj, transform);

    float4 aabb_min = float4(meshlet.aabb_min, 1.f);
    float4 aabb_extent = float4(meshlet.aabb_extent, 0.f);
    float4 aabb_max = aabb_min + aabb_extent;
    Aabb aabb = { aabb_min, aabb_max };

    // Culling.
    if (!frustrum_cull(mvp, aabb)) {
        return;
    }
    if (!cone_cull(mvp, meshlet.cone)) {
        return;
    }

    Command command;
    command.index_count = 372; // 124 * 3 - always 124 triangles per meshlet.
    command.instance_count = 1;
    command.first_index = meshlet_id * 372;
    command.vertex_offset = meshlet_id << 6; // Always 64 vertices per meshlet.
    command.first_instance = pointer.instance;

    u32 out_index = Constants.util.atomic_add(0, 1);
    Constants.commands.store(out_index, command);
}
