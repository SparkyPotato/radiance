import graph.types;
import asset;
import common;
import task;

[numthreads(64, 1, 1)]
[shader("amplification")]
public void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    if (id < Constants.meshlet_count) {
        MeshletPointer pointer = Constants.meshlet_pointers[id];
        Instance instance = Constants.instances[pointer.instance];
        Meshlet meshlet = ((Meshlet*)instance.mesh)[pointer.meshlet];
        Camera camera = Constants.camera[0];
        Camera prev_camera = Constants.camera[1];

        float4x4 transform = instance.get_transform();
        float4 sphere = float4(meshlet.bounding);
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.view_proj, transform);
        
        bool visible = decide_lod(mv, camera.h, float4(meshlet.group_error), float4(meshlet.parent_error));
        visible = visible && frustum_cull(mvp, sphere);
        if (visible) {
            visible = occlusion_cull(transform, prev_camera, sphere);
            if (!visible) {
                u32 did;
                InterlockedAdd(Constants.culled[0], 1, did);
                if ((did & 63) == 0) {
                    InterlockedAdd(Constants.culled[1], 1);
                }
                Constants.culled[4 + did] = id;
            }
        }

        if (visible) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index].pointer = pointer;
            Payload.pointers[index].id = id;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
