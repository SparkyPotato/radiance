import graph.types;
import asset;
import common;
import task;

[numthreads(64, 1, 1)]
[shader("amplification")]
public void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    if (id < Constants.culled[0]) {
        u32 pid = Constants.culled[4 + id];
        MeshletPointer pointer = Constants.meshlet_pointers[pid];
        Instance instance = Constants.instances[pointer.instance];
        Meshlet meshlet = ((Meshlet*)instance.mesh)[pointer.meshlet];
        Camera camera = *Constants.camera;

        float4 sphere = float4(meshlet.bounding);
        float4x4 transform = instance.get_transform();
        bool visible = occlusion_cull(transform, camera, sphere);

        if (visible) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index].pointer = pointer;
            Payload.pointers[index].id = pid;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
