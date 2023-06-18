#include "radiance-passes/mesh/data.l.hlsl"

#include "common.l.hlsl"

struct PushConstants {
    Buf<Meshlet> meshlets;
    Buf<Instance> instances;
    Buf<Vertex> vertices;
    Buf<Camera> camera;
};

PUSH PushConstants Constants;

VertexOutput main(u32 vertex_id: SV_VertexID, u32 instance_id: SV_InstanceID) {
    Instance instance = Constants.instances.load(instance_id);
    u32 meshlet_id = vertex_id >> 6; // Always 64 vertices per meshlet.
    Meshlet meshlet = Constants.meshlets.load(meshlet_id);
    Camera camera = Constants.camera.load(0);
    Vertex vertex = Constants.vertices.load(vertex_id);

    float3 normalized = float3(vertex.position) / 65535.0;
    float3 aabb_min = float3(meshlet.aabb_min);
    float3 aabb_extent = float3(meshlet.aabb_extent);
    float3 meshlet_pos = aabb_min + aabb_extent * normalized;

    float t[12] = instance.transform;
    float4x4 transform = {
        t[0], t[3], t[6], t[9],
        t[1], t[4], t[7], t[10],
        t[2], t[5], t[8], t[11],
        0.f, 0.f, 0.f, 1.f,
    };
    float4x4 mv = mul(camera.view, transform);
    float4x4 mvp = mul(camera.proj, mv);
    float4 position = mul(mvp, float4(meshlet_pos, 1.f));

    VertexOutput ret = { position, meshlet_id };
    return ret;
}
