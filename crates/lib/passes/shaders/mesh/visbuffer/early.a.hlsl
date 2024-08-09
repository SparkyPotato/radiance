#include "task.l.hlsl"

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    if (id < Constants.meshlet_count) {
        MeshletPointer pointer = Constants.meshlet_pointers.load(id);
        Instance instance = Constants.instances.load(pointer.instance);
        Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
        Camera camera = Constants.camera.load(0);

        float4 sphere = float4(meshlet.bounding);
        float4x4 transform = instance.get_transform();
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.view_proj, transform);
        
        bool visible = decide_lod(mv, camera.h, float4(meshlet.group_error), float4(meshlet.parent_error));
        visible = visible && frustum_cull(mvp, sphere);
        if (visible) {
            if (occlusion_cull(Constants.camera.load(1), transform, sphere)) {
                u32 index;
                InterlockedAdd(MeshletEmitCount, 1, index);
                Payload.pointers[index].pointer = pointer;
                Payload.pointers[index].id = id;
            } else {
                u32 did = Constants.culled.atomic_add(0, 1);
                Constants.culled.store(4 + did, id);
                if ((did & 63) == 0) {
                    Constants.culled.atomic_add(1, 1);
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
