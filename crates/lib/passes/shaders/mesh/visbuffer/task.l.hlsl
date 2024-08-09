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

bool occlusion_cull(Camera camera, float4x4 transform, float4 sphere) {
    float4x4 mv = mul(camera.view, transform); 
    float4 s = transform_sphere(mv, sphere);
    if (s.z < s.w + camera.near) return true;

    float3 cr = s.xyz * s.w;
    f32 czr2 = s.z * s.z - s.w * s.w;

    f32 vx = sqrt(s.x * s.x + czr2);
    f32 minx = (vx * s.x - cr.z) / (vx * s.z + cr.x);
    f32 maxx = (vx * s.x + cr.z) / (vx * s.z - cr.x);

    f32 vy = sqrt(s.y * s.y + czr2);
    f32 miny = (vy * s.y - cr.z) / (vy * s.z + cr.y);
    f32 maxy = (vy * s.y + cr.z) / (vy * s.z - cr.y);

    float4 aabb = float4(minx * camera.w, miny * camera.h, maxx * camera.w, maxy * camera.h);
    aabb = aabb.xwzy * float4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;

    f32 width = (aabb.z - aabb.x) * Constants.width / 2;
    f32 height = (aabb.w - aabb.y) * Constants.height / 2;
    f32 level = ceil(log2(max(width, height)));
    f32 depth = Constants.hzb.sample_mip(Constants.hzb_sampler, (aabb.xy + aabb.zw) * 0.5f, level).x;
    f32 closest = camera.near / (s.z - s.w);
    return closest >= depth;
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
