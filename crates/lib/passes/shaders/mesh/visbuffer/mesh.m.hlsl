#include "radiance-passes/asset/data.l.hlsl"
#include "visbuffer.l.hlsl"

struct VertexOutput {
    float4 position: SV_Position;
};

struct PrimitiveOutput {
    [[vk::location(0)]] u32 data: VisBuffer;
};

struct PushConstants {
    Buf<bytes> instances;
    Buf<Camera> camera;
    Buf<u32> i;
    u32 instance_count;
};

PUSH PushConstants Constants;

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void main(
    u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
    out vertices VertexOutput vertices[64],
    out indices uint3 triangles[124],
    out primitives PrimitiveOutput visbuffer[124]
) {
    u32 id = Constants.i.load(3 + gid);
    MeshletPointer pointer = MeshletPointer::get(Constants.instances, Constants.instance_count, id);
    Instance instance = Instance::get(Constants.instances, Constants.instance_count, pointer.instance);
    MeshletData meshlet = MeshletData::get(instance.mesh, pointer.meshlet_count, pointer.meshlet);
    Camera camera = Constants.camera.load(0);

    u32 vert_count = meshlet.vertex_count();
    u32 tri_count = meshlet.tri_count();
    SetMeshOutputCounts(vert_count, tri_count);

    float4x4 mvp = mul(camera.view_proj, instance.get_transform());

    if (gtid < vert_count) {
        Vertex vertex = meshlet.vertex(instance.mesh, gtid);
        vertices[gtid].position = mul(mvp, float4(vertex.position, 1.f));
    }

    for (u32 t = gtid; t < tri_count; t += 64) {
        u32 indices = meshlet.tri(instance.mesh, t);
        triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

        VisBufferData data = { id, t };
        visbuffer[t].data = data.encode();
    }
}
