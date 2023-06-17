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

bool frustrum_cull(float4x4 view_proj, float4 aabb_min, float4 aabb_max) {
    float4 aabb_min_clip = mul(view_proj, aabb_min);
    float4 aabb_max_clip = mul(view_proj, aabb_max);

    float4 aabb_min_ndc = aabb_min_clip / aabb_min_clip.w;
    float4 aabb_max_ndc = aabb_max_clip / aabb_max_clip.w;

    return (aabb_min_clip.w > 0.f && aabb_max_clip.w > 0.f)
        && (aabb_min_ndc.x <= 1.f && aabb_max_ndc.x >= -1.f)
        && (aabb_min_ndc.y <= 1.f && aabb_max_ndc.y >= -1.f)
        && (aabb_min_ndc.z <= 1.f && aabb_max_ndc.z >= 0.f);
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

    // Frustrum culling.
    float4 aabb_min = mul(transform, float4(meshlet.aabb_min, 1.0));
    float4 aabb_extent = float4(meshlet.aabb_extent, 0.0);
    float4 aabb_max = mul(transform, aabb_min + aabb_extent);

    if (!frustrum_cull(camera.view_proj, aabb_min, aabb_max)) {
        return;
    }

    u32 out_index = Constants.util.atomic_add(0, 1);
    Command command;

    command.index_count = 372; // 124 * 3 - always 124 triangles per meshlet.
    command.instance_count = 1;
    command.first_index = meshlet_id * 372;
    command.vertex_offset = meshlet_id << 6; // Always 64 vertices per meshlet.
    command.first_instance = pointer.instance;

    Constants.commands.store(out_index, command);
}
