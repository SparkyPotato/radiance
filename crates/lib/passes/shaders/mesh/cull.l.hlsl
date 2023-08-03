#pragma once

struct Sphere {
    float3 center;
    float radius;
};

struct Aabb {
    float4 min;
    float4 extent;
    float4 max;

    Sphere get_sphere() {
        float3 half_extent = this.extent.xyz * 0.5f;
        Sphere ret;
        ret.center = this.min.xyz + half_extent;
        ret.radius = length(half_extent);
        return ret;
    }
};

struct Cone {
    u32 apex;
    u32 axis_cutoff;
};

bool frustum_cull(float4 frustum, f32 near, float4x4 mv, Aabb aabb) {
    float scale = max(max(abs(mv._m00), abs(mv._m11)), abs(mv._m22));

    Sphere sphere = aabb.get_sphere();
    float3 center = mul(mv, sphere.center);
    float radius = sphere.radius * scale;

    bool visible = true;

    visible = visible && center.z * frustum.y - abs(center.x) * frustum.x > -radius;
    visible = visible && center.z * frustum.w - abs(center.y) * frustum.z > -radius;
    visible = visible && center.z + near > -radius;

    return visible;
}

bool cone_cull(float4x4 mv, Aabb aabb, Cone cone) {
    u16 apex_x = u16((cone.apex >> 0) & 0xff);
    u16 apex_y = u16((cone.apex >> 8) & 0xff);
    u16 apex_z = u16((cone.apex >> 16) & 0xff);
    float3 apex_normalized = float3(apex_x, apex_y, apex_z) / 255.f;
    float4 apex = aabb.min + aabb.extent * float4(apex_normalized, 0.f);

    u16 axis_x = u16((cone.axis_cutoff >> 0) & 0xff);
    u16 axis_y = u16((cone.axis_cutoff >> 8) & 0xff);
    u16 axis_z = u16((cone.axis_cutoff >> 16) & 0xff);
    u16 int_cutoff = u16((cone.axis_cutoff >> 24) & 0xff);

    vector<u16, 4> int_cone = { axis_x, axis_y, axis_z, int_cutoff };
    vector<u16, 4> sign = (int_cone & 0x80) >> 7;
    u16 signext = 0xff << 8;
    vector<i16, 4> signed_cone = vector<i16, 4>((sign * signext) | int_cone);

    float4 norm_cone = float4(signed_cone) / 127.f;
    float3 axis = normalize(norm_cone.xyz);
    float cutoff = norm_cone.w;

    float3 apex_camera = mul(mv, apex).xyz;
    float3 axis_camera = mul(mv, float4(axis, 0.f)).xyz;

    return dot(normalize(apex_camera), normalize(axis_camera)) >= cutoff;
}
