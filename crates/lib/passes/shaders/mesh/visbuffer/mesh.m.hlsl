#include "common.l.hlsl"
#include "visbuffer.l.hlsl"

PUSH PushConstants Constants;

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void main(
    u32 gid: SV_GroupID, u32 gtid: SV_GroupThreadID,
    in payload MeshPayload payload,
    out vertices VertexOutput vertices[64],
    out indices uint3 triangles[124],
    out primitives PrimitiveOutput visbuffer[124]
) {
    PointerWithId pointer_id = payload.pointers[gid];
    MeshletPointer pointer = pointer_id.pointer;
    u32 id = pointer_id.id;

    Instance instance = Constants.instances.load(pointer.instance);
    Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
    Camera camera = Constants.camera.load(0);

    u32 vert_count = (meshlet.vert_and_tri_count >> 0) & 0xff;
    u32 tri_count = (meshlet.vert_and_tri_count >> 8) & 0xff;
    SetMeshOutputCounts(vert_count, tri_count);

    // 64 threads per group and upto 64 vertices per meshlet.
    if (gtid < vert_count) {
        Vertex vertex = instance.mesh.load<Vertex>(meshlet.vertex_byte_offset, gtid);

        float3 normalized = float3(vertex.position) / 65535.0;
        float3 aabb_min = float3(meshlet.aabb_min);
        float3 aabb_extent = float3(meshlet.aabb_extent);
        float3 meshlet_pos = aabb_min + aabb_extent * normalized;

        float t[12] = instance.transform;
        float4x4 transform = {
            t[0], t[3], t[6], t[9],
            t[1], t[4], t[7], t[10],
            t[2], t[5], t[8], t[11],
            0.f,  0.f,  0.f,  1.f,
        };
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.proj, mv);
        float4 position = mul(mvp, float4(meshlet_pos, 1.f));

        vertices[gtid].position = position;
    }

    for (u32 t = gtid; t < tri_count; t += 64) {
        u32 indices = instance.mesh.load<u32>(meshlet.index_byte_offset, t);
        triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

        VisBufferData data = { id, t };
        visbuffer[t].data = data.encode();
    }
}
