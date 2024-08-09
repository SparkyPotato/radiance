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

    if (id < Constants.culled[0]) {
        let pid = Constants.culled[4 + id];
        let pointer = Constants.meshlet_pointers[pid];
        let instance = Constants.instances[pointer.instance];
        let meshlet = ((Meshlet*)instance.mesh)[pointer.meshlet];
        let camera = *Constants.camera;

        let transform = instance.get_transform();
        let visible = occlusion_cull(transform, camera, meshlet.bounding);

        if (visible) {
            u32 i = 1;
            u32 index = spirv_asm { result:$$u32 = OpAtomicIAdd &MeshletEmitCount Workgroup None $i };
            // InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index] = { pointer, id };
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
