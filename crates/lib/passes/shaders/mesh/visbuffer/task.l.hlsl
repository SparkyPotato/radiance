#include "common.l.hlsl"

PUSH PushConstants Constants;

groupshared u32 MeshletEmitCount;
groupshared MeshPayload Payload;

float4 normalize_plane(float4 p) {
    float3 n = p.xyz;
    float l = length(n);
    return p / l;
}

struct Frustum {
    float4 planes[5];

    static Frustum from_matrix(float4x4 p) {
        Frustum ret;

        ret.planes[0] = normalize_plane(p[3] + p[0]);
        ret.planes[1] = normalize_plane(p[3] - p[0]);
        ret.planes[2] = normalize_plane(p[3] + p[1]);
        ret.planes[3] = normalize_plane(p[3] - p[1]);
        ret.planes[4] = normalize_plane(p[2]);

        return ret;
    }
};

f32 plane_distance(float4 plane, float3 p) {
    return dot(float4(p, 1.f), plane);
}

bool frustum_cull(float4x4 mvp, float4 sphere) {
    Frustum f = Frustum::from_matrix(mvp);

    f32 dist = plane_distance(f.planes[0], sphere.xyz);
    for (int i = 1; i < 5; i++) {
        dist = min(dist, plane_distance(f.planes[i], sphere.xyz));
    }
    return dist > -sphere.w;
}

float4 transform_sphere(float4x4 mv, float4 sphere) {
    float scale = max(max(length(mv._m00_m10_m20), length(mv._m01_m11_m21)), length(mv._m02_m12_m22));
    float4 center = mul(mv, float4(sphere.xyz, 1.f));
    return float4(center.xyz, sphere.w * scale * 0.5f);
}

bool occlusion_cull(float4x4 mv, float w, float h, float near, float4 sphere) {
    float4 s = transform_sphere(mv, sphere);
    if (s.z < s.w + near) return true;

    float3 cr = s.xyz * s.w;
    f32 czr2 = s.z * s.z - s.w * s.w;
    f32 vx = sqrt(s.x * s.x + czr2);
    f32 minx = (vx * s.x - cr.z) / (vx * s.z + cr.x);
    f32 maxx = (vx * s.x + cr.z) / (vx * s.z - cr.x);
    f32 vy = sqrt(s.y * s.y + czr2);
    f32 miny = (vy * s.y - cr.z) / (vy * s.z + cr.y);
    f32 maxy = (vy * s.y + cr.z) / (vy * s.z - cr.y);
    float4 aabb = float4(minx * w, miny * h, maxx * w, maxy * h);
    aabb = aabb.xwzy * float4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;

    f32 width = (aabb.z - aabb.x) * Constants.width / 2;
    f32 height = (aabb.w - aabb.y) * Constants.height / 2;
    f32 level = ceil(log2(max(width, height)));
    f32 depth = Constants.hzb.sample_mip(Constants.hzb_sampler, (aabb.xy + aabb.zw) * 0.5f, level).x;
    f32 closest = near / (s.z - s.w);
    return closest > depth;
}

f32 is_imperceptible(float4x4 mv, float h, float4 error) {
    float4 sphere = transform_sphere(mv, error);
    f32 d2 = dot(sphere.xyz, sphere.xyz);
    f32 r2 = sphere.w * sphere.w;
    f32 dia = h * sphere.w / sqrt(d2 - r2);
    return dia * f32(max(Constants.width, Constants.height)) < 1.f;
}

bool decide_lod(float4x4 mv, float h, float4 group_error, float4 parent_error) {
    return is_imperceptible(mv, h, group_error) && !is_imperceptible(mv, h, parent_error);
}

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupThreadID) {
    if (gtid == 0) { MeshletEmitCount = 0; }
    GroupMemoryBarrierWithGroupSync();

    u32 meshlet_count = 
    #ifdef EARLY
    Constants.meshlet_count;
    #else
    Constants.culled.load(0);
    #endif
    if (id < meshlet_count) {
        u32 pointer_id =
        #ifdef EARLY
        id;
        #else
        Constants.culled.load(4 + id);
        #endif
        MeshletPointer pointer = Constants.meshlet_pointers.load(pointer_id);
        Instance instance = Constants.instances.load(pointer.instance);
        Meshlet meshlet = instance.mesh.load<Meshlet>(0, pointer.meshlet);
        Camera camera = Constants.camera.load(0);
        Camera prev_camera = Constants.camera.load(1);

        float4x4 transform = instance.get_transform();
        float4 sphere = float4(meshlet.bounding);
        float4x4 mv = mul(camera.view, transform);
        float4x4 mvp = mul(camera.view_proj, transform);
        float4x4 pmv = mul(prev_camera.view, transform);
        float4x4 pmvp = mul(prev_camera.view_proj, transform);
        
        #ifdef EARLY
        bool visible = decide_lod(mv, camera.h, float4(meshlet.group_error), float4(meshlet.parent_error));
        visible = visible && frustum_cull(mvp, sphere);
        #else
        bool visible = true;
        #endif

        if (visible) {
            #ifdef EARLY
            bool visible_last_frame = occlusion_cull(pmv, prev_camera.w, prev_camera.h, prev_camera.near, sphere);
            if (!visible_last_frame) {
                u32 did = Constants.culled.atomic_add(0, 1);
                if ((did & 63) == 0) {
                    Constants.culled.atomic_add(1, 1);
                }
                Constants.culled.store(4 + did, pointer_id);
            }
            visible = visible_last_frame;
            #else
            visible = occlusion_cull(mv, camera.w, camera.h, camera.near, sphere);
            #endif
        }

        if (visible) {
            u32 index;
            InterlockedAdd(MeshletEmitCount, 1, index);
            Payload.pointers[index].pointer = pointer;
            Payload.pointers[index].id = pointer_id;
        }
    }

    // Emit.
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(MeshletEmitCount, 1, 1, Payload);
}
