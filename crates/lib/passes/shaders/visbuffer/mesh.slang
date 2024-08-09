import graph.types;
import asset;
import common;
import visbuffer;

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
[shader("mesh")]
public void main(
    u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
    in payload MeshPayload payload,
    out vertices VertexOutput vertices[64],
    out indices uint3 triangles[124],
    out primitives PrimitiveOutput visbuffer[124]
) {
    let pointer_id = payload.pointers[gid];
    let pointer = pointer_id.pointer;
    let id = pointer_id.id;

    let instance = Constants.instances[pointer.instance];
    let meshlet = ((Meshlet*)instance.mesh)[pointer.meshlet];
    let camera = *Constants.camera;

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    let transform = instance.get_transform();
    let mvp = mul(camera.view_proj, transform);
    let vertices = (Vertex*)(instance.mesh + meshlet.vertex_offset);
    let indices = (u32*)(instance.mesh + meshlet.index_offset);

    // 64 threads per group and upto 64 vertices per meshlet.
    if (gtid < meshlet.vertex_count) {
        let vertex = vertices[gtid];
        vertices[gtid].position = mul(mvp, f32x4(f32x3(vertex.position), 1.f)).xyz;
    }

    for (int t = gtid; t < meshlet.triangle_count; t += 64) {
        let indices = indices[t];
        triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

        VisBufferData data = { id, t };
        visbuffer[t].data = data.encode();
    }
}
