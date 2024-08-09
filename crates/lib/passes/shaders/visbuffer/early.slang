import graph.types;
import asset;
import common;
import task;

groupshared u32 MeshletEmitCount;
groupshared MeshPayload Payload;

[numthreads(64, 1, 1)]
[shader("amplification")]
public void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    if (id < Constants.meshlet_count) {
        let pointer = Constants.meshlet_pointers[id];
        let instance = Constants.instances[pointer.instance];
        let meshlet = ((Meshlet*)instance.mesh)[pointer.meshlet];
        let camera = Constants.camera[0];
        let prev_camera = Constants.camera[1];

        let transform = instance.get_transform();
        let mv = mul(camera.view, transform);
        let mvp = mul(camera.view_proj, transform);
        
        var visible = decide_lod(mv, camera.h, meshlet.group_error, meshlet.parent_error);
        visible = visible && frustum_cull(mvp, meshlet.bounding);
        if (visible) {
            visible = occlusion_cull(transform, prev_camera, meshlet.bounding);
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
            u32 i = 1;
            u32 index = spirv_asm { result:$$u32 = OpAtomicIAdd &MeshletEmitCount Workgroup None $i };
            // u32 index;
            // InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index] = { id };
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
