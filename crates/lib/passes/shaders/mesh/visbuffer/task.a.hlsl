#include "common.l.hlsl"

PUSH PushConstants Constants;

groupshared u32 MeshletEmitCount = 0;
groupshared MeshPayload Payload;

[numthreads(64, 1, 1)]
void main(uint id: SV_DispatchThreadID, uint gtid: SV_GroupThreadID) {
    if (id >= Constants.meshlet_count) {
        return;
    }

    MeshletPointer pointer = Constants.meshlet_pointers.load(id);
    Instance instance = Constants.instances.load(pointer.instance);
    Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
    Camera camera = Constants.camera.load(0);

    float4x4 transform = instance.get_transform();
    float4x4 mv = mul(camera.view, transform);
    Aabb aabb = meshlet.get_mesh_aabb();

    // Culling.
    bool culled = false;
    culled = culled || frustum_cull(camera.frustum, camera.near, mv, aabb);
    // culled = culled || cone_cull(mv, aabb, meshlet.cone);

    // Write appropriate meshlet id to the payload.
    if (!culled) {
        InterlockedAdd(MeshletEmitCount, 1);
        Payload.pointers[gtid].pointer = pointer;
        Payload.pointers[gtid].id = id;
    }

    // Emit.
    AllMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}