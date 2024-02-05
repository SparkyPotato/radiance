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
        ret.planes[4] = normalize_plane(p[3] - p[2]);

        return ret;
    }
};

bool frustum_cull(float4x4 mvp, Aabb aabb) {
    Frustum frustum = Frustum::from_matrix(mvp);

    float3 half_extent = aabb.extent.xyz * 0.5f;
    float3 center = aabb.min.xyz + half_extent;

    for (u32 i = 0; i < 5; i++) {
        float4 p = frustum.planes[i];
        float3 plane = frustum.planes[i].xyz;
        float3 abs_plane = abs(plane);

        float d = dot(center, plane);
        float r = dot(half_extent, abs_plane);

        if (d + r > -p.w) {
            return false;
        }
    }

    return true;
}
