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

        float4x4 transform = instance.get_transform();
        float4x4 mv = mul(camera.view, transform);
        float4 sphere = transform_sphere(mv, float4(meshlet.bounding));
        float4 group_error = transform_sphere(mv, float4(meshlet.group_error));
        float4 parent_error = transform_sphere(mv, float4(meshlet.parent_error));
        
        bool visible = decide_lod(camera.h, group_error, parent_error);
        visible = visible && frustum_cull(camera.frustum, camera.near, sphere);
        if (visible) {
            if (occlusion_cull(Constants.camera.load(1), transform, float4(meshlet.bounding))) {
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
