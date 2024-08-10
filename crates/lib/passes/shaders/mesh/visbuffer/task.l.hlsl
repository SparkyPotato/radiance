#include "common.l.hlsl"

PUSH PushConstants Constants;

groupshared u32 MeshletEmitCount;
groupshared MeshPayload Payload;

f32 is_imperceptible(float h, float4 error) {
    f32 d2 = dot(error.xyz, error.xyz);
    f32 r2 = error.w * error.w;
    f32 dia = h * error.w / sqrt(d2 - r2);
    return dia * f32(max(Constants.width, Constants.height)) < 1.f;
}

bool decide_lod(float h, float4 group_error, float4 parent_error) {
    return is_imperceptible(h, group_error) && !is_imperceptible(h, parent_error);
}
bool frustum_cull(float4 f, float near, float4 sphere) {
    bool visible = true;
    visible = visible && sphere.z * f.y - abs(sphere.x) * f.x > -sphere.w;
    visible = visible && sphere.z * f.w - abs(sphere.y) * f.z > -sphere.w;
    visible = visible && sphere.z + sphere.w > near;
    return visible;
}

float4 transform_sphere(float4x4 mv, float4 sphere) {
    float scale = max(max(length(mv._m00_m10_m20), length(mv._m01_m11_m21)), length(mv._m02_m12_m22));
    float4 center = mul(mv, float4(sphere.xyz, 1.f));
    return float4(center.xyz, sphere.w * scale);
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

    f32 w = camera.w * 0.5f;
    f32 h = -camera.h * 0.5f;
    float4 aabb = float4(minx * w, maxy * h, maxx * w, miny * h) + 0.5f;

    f32 width = (aabb.z - aabb.x) * f32(Constants.width) / 2.f;
    f32 height = (aabb.w - aabb.y) * f32(Constants.height) / 2.f;
    f32 level = ceil(log2(max(width, height)));
    f32 depth = Constants.hzb.sample_mip(Constants.hzb_sampler, (aabb.xy + aabb.zw) * 0.5f, level).x;
    f32 closest = camera.near / (s.z - s.w);
    return closest >= depth;
}
