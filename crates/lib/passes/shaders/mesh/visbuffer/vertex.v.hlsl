#include "radiance-passes/mesh/data.l.hlsl"

#include "common.l.hlsl"

struct PushConstants {
    Buf<Vertex> vertices;
    Buf<Meshlet> meshlets;
    Buf<Camera> camera;
};

PUSH PushConstants Constants;

VertexOutput main(u32 vertex_id: SV_VertexID, u32 meshlet_id: SV_InstanceID) {
    Meshlet meshlet = Constants.meshlets.load(meshlet_id - 1);
    Camera camera = Constants.camera.load(0);
    Vertex vertex = Constants.vertices.load(vertex_id);

    float3 pos = float3(vertex.position[0], vertex.position[1], vertex.position[2]);
    float4 normalized = float4(float3(vertex.position) / 65535.0, 1.0);
    float t[12] = meshlet.transform;
    float4x4 transform = {
        t[0], t[3], t[6], t[9],
        t[1], t[4], t[7], t[10],
        t[2], t[5], t[8], t[11],
        0.f, 0.f, 0.f, 1.f,
    };
    float4x4 mvp = mul(camera.view_proj, transform);
    float4 position = mul(mvp, normalized);

    VertexOutput ret = { position, meshlet_id };
    return ret;
}
