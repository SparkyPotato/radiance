#include "common.l.hlsl"

PUSH PushConstants Constants;

groupshared u32 MeshletEmitCount = 0;
groupshared MeshPayload Payload;

void write_pointer(u32 off, u32 doff, u32 mpid) {
    u32 i = Constants.ww.atomic_add(off, 1);
    Constants.wd.store(doff + i, mpid);
    if (i % 64 == 0) {
        Constants.ww.atomic_add(off + 1, 1);
    }
}

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    u32 count = Constants.rw.load(WORKGROUP_OFFSET);
    if (id < count) {
        u32 mpid = Constants.rd.load(DATA_OFFSET + id);
        MeshletPointer pointer = Constants.meshlet_pointers.load(mpid);
        Instance instance = Constants.instances.load(pointer.instance);
        Meshlet meshlet = instance.mesh.load<Meshlet>(sizeof(u32) * instance.submesh_count, pointer.meshlet);
        Camera camera = Constants.camera.load(CULL_CAMERA);

        float4x4 transform = instance.get_transform();
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.view_proj, transform);
        Aabb aabb = meshlet.get_mesh_aabb();

        // Culling.
        bool culled = frustum_cull(mvp, aabb);

        // Write appropriate meshlet id to the payload.
        u32 wo = INVISIBLE_WORKGROUP_OFFSET;
        u32 doff = INVISIBLE_DATA_OFFSET;
        if (!culled) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index].pointer = pointer;
            Payload.pointers[index].id = mpid;

            wo = VISIBLE_WORKGROUP_OFFSET;
            doff = VISIBLE_DATA_OFFSET;
        }
        write_pointer(wo, doff, mpid);
    }

    // Emit.
    AllMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
