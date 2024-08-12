#define LATE
#include "cull.l.hlsl"

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    u32 meshlet_count = Constants.culled.load(0);
    if (id >= meshlet_count) return;

    u32 pointer_id = Constants.culled.load(4 + id);
    MeshletPointer pointer = MeshletPointer::get(Constants.instances, Constants.instance_count, pointer_id);
    Instance instance = Instance::get(Constants.instances, Constants.instance_count, pointer.instance);
    MeshletBounds meshlet = MeshletBounds::get(instance.mesh, pointer.meshlet);
    float4x4 transform = instance.get_transform();

    if (occlusion_cull(Constants.camera.load(0), transform, meshlet.bounding)) {
        u32 did = Constants.o.atomic_add(0, 1);
        Constants.o.store(3 + did, pointer_id);
    }
}
