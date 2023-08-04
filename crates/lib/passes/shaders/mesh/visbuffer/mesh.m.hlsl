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
    Camera camera = Constants.camera.load(DRAW_CAMERA);

    u32 vert_count = (meshlet.vert_and_tri_count >> 0) & 0xff;
    u32 tri_count = (meshlet.vert_and_tri_count >> 8) & 0xff;
    SetMeshOutputCounts(vert_count, tri_count);

    float4x4 transform = instance.get_transform();
    float4x4 mvp = mul(camera.view_proj, transform);
    Aabb aabb = meshlet.get_mesh_aabb();

    // 64 threads per group and upto 64 vertices per meshlet.
    if (gtid < vert_count) {
        Vertex vertex = instance.mesh.load<Vertex>(meshlet.vertex_byte_offset, gtid);

        float4 normalized = float4(vertex.position, 0.f) / 65535.0;
        float4 pos = aabb.min + aabb.extent * normalized;
        vertices[gtid].position = mul(mvp, pos);
    }

    for (u32 t = gtid; t < tri_count; t += 64) {
        u32 indices = instance.mesh.load<u32>(meshlet.index_byte_offset, t);
        triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

        VisBufferData data = { id, t };
        visbuffer[t].data = data.encode();
    }
}

[outputtopology("line")]
[numthreads(64, 1, 1)]
void old_main(
    u32 gid: SV_GroupID, u32 gtid: SV_GroupThreadID,
    in payload MeshPayload payload,
    out vertices VertexOutput vertices[8],
    out indices uint2 lines[12],
    out primitives PrimitiveOutput visbuffer[12]
) {
    PointerWithId pointer_id = payload.pointers[gid];
    MeshletPointer pointer = pointer_id.pointer;
    u32 id = pointer_id.id;

    Instance instance = Constants.instances.load(pointer.instance);
    Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
    Camera camera = Constants.camera.load(0);

    float4x4 transform = instance.get_transform();
    float4x4 mvp = mul(camera.view_proj, transform);
    Aabb aabb = meshlet.get_mesh_aabb();

    SetMeshOutputCounts(8, 12);

    if (gtid < 8) {
        // Select aabb corner
        float4 pos = float4(
            (gtid & 1) ? aabb.min.x : aabb.max.x,
            (gtid & 2) ? aabb.min.y : aabb.max.y,
            (gtid & 4) ? aabb.min.z : aabb.max.z,
            1.f
        );

        float4 position = mul(mvp, pos);
        vertices[gtid].position = position;
    }

    if (gtid < 12) {
        // Generate edge lines.
        uint2 o;
        switch (gtid) {
            case 0: o = uint2(0, 1); break;
            case 1: o = uint2(0, 2); break;
            case 2: o = uint2(0, 4); break;
            case 3: o = uint2(1, 3); break;
            case 4: o = uint2(1, 5); break;
            case 5: o = uint2(2, 3); break;
            case 6: o = uint2(2, 6); break;
            case 7: o = uint2(3, 7); break;
            case 8: o = uint2(4, 5); break;
            case 9: o = uint2(4, 6); break;
            case 10: o = uint2(5, 7); break;
            case 11: o = uint2(6, 7); break;
        }
        lines[gtid] = o;
        VisBufferData data = { id, gtid };
        visbuffer[gtid].data = data.encode();
    }
}
