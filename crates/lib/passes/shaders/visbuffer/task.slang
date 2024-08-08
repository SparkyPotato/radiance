import graph.types;
import asset;
import common;

public groupshared u32 MeshletEmitCount;
public groupshared MeshPayload Payload;

public func normalize_plane(p: f32x4) -> f32x4 {
    float3 n = p.xyz;
    float l = length(n);
    return p / l;
}

public struct Frustum {
    public f32x4 planes[5];

    public __init(float4x4 p) {
        Frustum ret;

        ret.planes[0] = normalize_plane(p[3] + p[0]);
        ret.planes[1] = normalize_plane(p[3] - p[0]);
        ret.planes[2] = normalize_plane(p[3] + p[1]);
        ret.planes[3] = normalize_plane(p[3] - p[1]);
        ret.planes[4] = normalize_plane(p[2]);

        return ret;
    }
}

public func plane_distance(plane: f32x4, p: f32x3) -> f32 {
    return dot(float4(p, 1.f), plane);
}

public func frustum_cull(mvp: float4x4, sphere: f32x4) -> bool {
    let f = Frustum(mvp);
    var dist = plane_distance(f.planes[0], sphere.xyz);
    for (int i = 1; i < 5; i++) {
        dist = min(dist, plane_distance(f.planes[i], sphere.xyz));
    }
    return dist > -sphere.w;
}

public func transform_sphere(mv: float4x4, sphere: f32x4) -> f32x4 {
    let scale = max(max(length(mv._m00_m10_m20), length(mv._m01_m11_m21)), length(mv._m02_m12_m22));
    let center = mul(mv, float4(sphere.xyz, 1.f));
    return f32x4(center.xyz, sphere.w * scale * 0.5f);
}

public func occlusion_cull(transform: float4x4, cam: Camera, sphere: f32x4) -> bool {
    let mv = mul(cam.view, transform);
    let s = transform_sphere(mv, sphere);
    if (s.z < s.w + cam.near) return true;

    let cr = s.xyz * s.w;
    let czr2 = s.z * s.z - s.w * s.w;
    let vx = sqrt(s.x * s.x + czr2);
    let minx = (vx * s.x - cr.z) / (vx * s.z + cr.x);
    let maxx = (vx * s.x + cr.z) / (vx * s.z - cr.x);
    let vy = sqrt(s.y * s.y + czr2);
    let miny = (vy * s.y - cr.z) / (vy * s.z + cr.y);
    let maxy = (vy * s.y + cr.z) / (vy * s.z - cr.y);
    var aabb = float4(minx * cam.w, miny * cam.h, maxx * cam.w, maxy * cam.h);
    aabb = aabb.xwzy * float4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;

    let width = (aabb.z - aabb.x) * Constants.width / 2;
    let height = (aabb.w - aabb.y) * Constants.height / 2;
    let level = ceil(log2(max(width, height)));
    let depth = Constants.hzb.sample_mip(Constants.hzb_sampler, (aabb.xy + aabb.zw) * 0.5f, level).x;
    let closest = cam.near / (s.z - s.w);
    return closest > depth;
}

public func is_imperceptible(mv: float4x4, h: f32, error: f32x4) -> bool  {
    let sphere = transform_sphere(mv, error);
    let d2 = dot(sphere.xyz, sphere.xyz);
    let r2 = sphere.w * sphere.w;
    let dia = h * sphere.w / sqrt(d2 - r2);
    return dia * f32(max(Constants.width, Constants.height)) < 1.f;
}

public func decide_lod(mv: float4x4, h: f32, group_error: f32x4, parent_error: f32x4) -> bool {
    return is_imperceptible(mv, h, group_error) && !is_imperceptible(mv, h, parent_error);
}
