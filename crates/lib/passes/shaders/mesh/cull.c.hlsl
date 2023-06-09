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
    float4 extent;
    float4 max;
};

Aabb transform_aabb(float4x4 transform, Aabb aabb) {
    float4 center = (aabb.min + aabb.max) * 0.5f;
    float4 half_extent = aabb.extent * 0.5f;

    float4 transformed_center = mul(transform, center);
    float3x3 abs_mat = float3x3(abs(transform[0].xyz), abs(transform[1].xyz), abs(transform[2].xyz));
    float4 transformed_half_extent = float4(mul(abs_mat, half_extent), 0.f);

    float4 transformed_min = transformed_center - transformed_half_extent;
    float4 transformed_max = transformed_center + transformed_half_extent;

    Aabb ret = { transformed_min, transformed_half_extent * 2.f, transformed_max };
    return ret;
}

bool frustrum_cull(float4x4 mvp, Aabb aabb) {
    Aabb clip = transform_aabb(mvp, aabb);
    float3 ndc_min = clip.min.xyz / clip.min.w;
    float3 ndc_max = clip.max.xyz / clip.max.w;

    return (clip.min.w <= 0.f || clip.max.w <= 0.f)
        || (ndc_min.x > 1.f || ndc_max.x < -1.f)
        || (ndc_min.y > 1.f || ndc_max.y < -1.f)
        || (ndc_min.z > 1.f || ndc_max.z < 0.f);
}

bool cone_cull(float4x4 mv, Aabb aabb, Cone cone) {
    u16 apex_x = u16((cone.apex >> 0) & 0xff);
    u16 apex_y = u16((cone.apex >> 8) & 0xff);
    u16 apex_z = u16((cone.apex >> 16) & 0xff);
    float3 apex_normalized = float3(apex_x, apex_y, apex_z) / 255.f;
    float4 apex = aabb.min + aabb.extent * float4(apex_normalized, 0.f);

    u16 axis_x = u16((cone.axis_cutoff >> 0) & 0xff);
    u16 axis_y = u16((cone.axis_cutoff >> 8) & 0xff);
    u16 axis_z = u16((cone.axis_cutoff >> 16) & 0xff);
    u16 int_cutoff = u16((cone.axis_cutoff >> 24) & 0xff);

    vector<u16, 4> int_cone = { axis_x, axis_y, axis_z, int_cutoff };
    vector<u16, 4> sign = (int_cone & 0x80) >> 7;
    u16 signext = 0xff << 8;
    vector<i16, 4> signed_cone = vector<i16, 4>((sign * signext) | int_cone);

    float4 norm_cone = float4(signed_cone) / 127.f;
    float3 axis = normalize(norm_cone.xyz);
    float cutoff = norm_cone.w;

    float3 apex_camera = mul(mv, apex).xyz;
    float3 axis_camera = mul(mv, float4(axis, 0.f)).xyz;

    return dot(normalize(apex_camera), normalize(axis_camera)) >= cutoff;
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
    float4x4 mv = mul(camera.view, transform);
    float4x4 mvp = mul(camera.proj, mv);

    float4 aabb_min = float4(meshlet.aabb_min, 1.f);
    float4 aabb_extent = float4(meshlet.aabb_extent, 0.f);
    float4 aabb_max = aabb_min + aabb_extent;
    Aabb aabb = { aabb_min, aabb_extent, aabb_max };

    // Culling.
    if (frustrum_cull(mvp, aabb)) {
        return;
    }
    if (cone_cull(mv, aabb, meshlet.cone)) {
        return;
    }

    Command command;
    command.index_count = 372; // 124 * 3 - always 124 triangles per meshlet.
    command.instance_count = 1;
    command.first_index = meshlet_id * 372;
    command.vertex_offset = meshlet_id << 6; // Always 64 vertices per meshlet.
    command.first_instance = index;

    u32 out_index = Constants.util.atomic_add(0, 1);
    Constants.commands.store(out_index, command);
}
