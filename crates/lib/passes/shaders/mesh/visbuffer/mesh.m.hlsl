#include "common.l.hlsl"
#include "visbuffer.l.hlsl"

PUSH PushConstants Constants;

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void main(
    u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
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
    Camera camera = Constants.camera.load(DRAW_CAMERA);

    u32 vert_count = (meshlet.vert_and_tri_count >> 0) & 0xff;
    u32 tri_count = (meshlet.vert_and_tri_count >> 8) & 0xff;
    SetMeshOutputCounts(vert_count, tri_count);

    float4x4 transform = instance.get_transform();
    float4x4 mvp = mul(camera.view_proj, transform);

    // 64 threads per group and upto 64 vertices per meshlet.
    if (gtid < vert_count) {
        Vertex vertex = instance.mesh.load<Vertex>(meshlet.vertex_offset, gtid);
        vertices[gtid].position = mul(mvp, float4(vertex.position, 1.f));
    }

    for (u32 t = gtid; t < tri_count; t += 64) {
        u32 indices = instance.mesh.load<u32>(meshlet.index_offset, t);
        triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

        VisBufferData data = { id, t };
        visbuffer[t].data = data.encode();
    }
}
