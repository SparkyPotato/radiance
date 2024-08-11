#include "cull.l.hlsl"

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { 
        MeshletEmitCount = 0;
        Payload.base = id;
    }
    GroupMemoryBarrierWithGroupSync();

    u32 meshlet_count = Constants.culled.load(0);
    if (id < meshlet_count) {
        u32 pointer_id = Constants.culled.load(4 + id);
        MeshletPointer pointer = Constants.meshlet_pointers.load(pointer_id);
        Instance instance = Constants.instances.load(pointer.instance);
        Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);

        float4 sphere = float4(meshlet.bounding);
        float4x4 transform = instance.get_transform(); 
        if (occlusion_cull(Constants.camera.load(0), transform, sphere)) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            payload_set(index, id);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
