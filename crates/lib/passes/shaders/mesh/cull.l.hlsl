#pragma once

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

float4 transform_sphere(float4x4 mvp, float4 sphere) {
    float scale = max(max(length(mvp._m00_m10_m20), length(mvp._m01_m11_m21)), length(mvp._m02_m12_m22));
    float4 center = mul(mvp, float4(sphere.xyz, 1.f));
    return float4(center.xyz, sphere.w * scale * 0.5f);
}

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
