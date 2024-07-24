#include "common.l.hlsl"

PUSH PushConstants Constants;

groupshared u32 MeshletEmitCount;
groupshared MeshPayload Payload;

f32 is_imperceptible(float4x4 mv, float cot_fov, float4 error) {
    float4 sphere = transform_sphere(mv, error);
    f32 d2 = dot(sphere.xyz, sphere.xyz);
    f32 r2 = sphere.w * sphere.w;
    f32 dia = (cot_fov * sphere.w) / sqrt(d2 - r2);
    return dia * f32(Constants.resolution) < 1.f;
}

bool decide_lod(float4x4 mv, float cot_fov, float4 group_error, float4 parent_error) {
    return is_imperceptible(mv, cot_fov, group_error) && !is_imperceptible(mv, cot_fov, parent_error);
}

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    if (id < Constants.meshlet_count) {
        MeshletPointer pointer = Constants.meshlet_pointers.load(id);
        Instance instance = Constants.instances.load(pointer.instance);
        Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
        Camera camera = Constants.camera.load(CULL_CAMERA);

        float4x4 transform = instance.get_transform();
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.view_proj, transform);

        // Culling.
        bool visible = decide_lod(mv, camera.cot_fov, float4(meshlet.group_error), float4(meshlet.parent_error));
        if (visible) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index].pointer = pointer;
            Payload.pointers[index].id = id;
        }
    }

    // Emit.
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
