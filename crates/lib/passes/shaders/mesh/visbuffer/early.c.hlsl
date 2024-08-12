#define EARLY
#include "cull.l.hlsl"

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (id >= Constants.meshlet_count) return;

    MeshletPointer pointer = MeshletPointer::get(Constants.instances, Constants.instance_count, id);
    Instance instance = Instance::get(Constants.instances, Constants.instance_count, pointer.instance);
    MeshletBounds meshlet = MeshletBounds::get(instance.mesh, pointer.meshlet);
    Camera camera = Constants.camera.load(0);

    float4x4 transform = instance.get_transform();
    float4x4 mv = mul(camera.view, transform);
    float4 sphere = transform_sphere(mv, meshlet.bounding);
    float4 group_error = transform_sphere(mv, meshlet.group_error);
    float4 parent_error = transform_sphere(mv, meshlet.parent_error);
        
    bool visible = decide_lod(camera.h, group_error, parent_error);
    visible = visible && frustum_cull(camera.frustum, camera.near, sphere);
    if (!visible) return;

    if (occlusion_cull(Constants.camera.load(1), transform, meshlet.bounding)) {
        u32 did = Constants.o.atomic_add(0, 1);
        Constants.o.store(3 + did, id);
    } else {
        u32 did = Constants.culled.atomic_add(0, 1);
        Constants.culled.store(4 + did, id);
        if ((did & 63) == 0) {
            Constants.culled.atomic_add(1, 1);
        }
    }
}
