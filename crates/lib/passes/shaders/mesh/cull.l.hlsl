#include "radiance-passes/asset/data.l.hlsl"

struct BvhQueue {
	Buf<bytes> buf;

	u32 len() {
		return this.buf.load<u32>(0, 0);
	}

	BvhNodePointer get(u32 id) {
		return this.buf.load<BvhNodePointer>(sizeof(u32) * 4, id);
	}

	void push(BvhNodePointer pointer) {
		u32 id = this.buf.atomic_add(0, 0, 1);
		this.buf.store(sizeof(u32) * 4, id, pointer);
		if ((id & 63) == 0) this.buf.atomic_add(0, 1, 1);
	}

	void push(BvhNodePointer pointer, u32 count, u32 size) {
		u32 base = this.buf.atomic_add(0, 0, count);
		for (u32 i = 0; i < count; i++) {
			this.buf.store(sizeof(u32) * 4, base + i, pointer);
			pointer.node += size;
		}
		u32 next_mul = ((base + 63) / 64) * 64;
		if (next_mul - base < count) this.buf.atomic_add(0, 1, 1);
	}
};

struct MeshletQueue {
	Buf<bytes> buf;

	u32 len() {
		return this.buf.load<u32>(0, 0);
	}

	MeshletPointer get(u32 id) {
		return this.buf.load<MeshletPointer>(sizeof(u32) * 3, id);
	}

	void push(MeshletPointer pointer) {
		u32 id = this.buf.atomic_add(0, 0, 1);
		this.buf.store(sizeof(u32) * 3, id, pointer);
	}
};

float4 normalize_plane(float4 p) {
	return p / length(p.xyz);
}

// https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
// https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
struct Cull {
	float4x4 mv;
	float2 screen;
	f32 h;
	float4 planes[5];
	float3 flips[5];

	static Cull init(float4x4 mv, float4x4 mvp, uint2 res, f32 h) {
		Cull ret;
		ret.mv = mv;
		ret.screen = float2(res);
		ret.h = h;
		ret.planes[0] = normalize_plane(mvp[3] + mvp[0]);
		ret.planes[1] = normalize_plane(mvp[3] - mvp[0]);
		ret.planes[2] = normalize_plane(mvp[3] + mvp[1]);
		ret.planes[3] = normalize_plane(mvp[3] - mvp[1]);
		ret.planes[4] = normalize_plane(mvp[2]);
		[unroll]
		for (int i = 0; i < 5; i++) {
			ret.flips[i] = sign(ret.planes[i].xyz);
		}
		return ret;
	} 

	bool frustum_cull(Aabb aabb) {
		[unroll]
		for (int i = 0; i < 5; i++) {
			float4 plane = this.planes[i];
			float3 flip = this.flips[i];
			if (dot(aabb.center + aabb.half_extent * flip, plane.xyz) > -plane.w) return false;
		}
		return true;
	}

	float4 transform_sphere(float4 sphere) {
		float3 x = this.mv._m00_m10_m20;
		float3 y = this.mv._m01_m11_m21;
		float3 z = this.mv._m02_m12_m22;
		f32 m = max(dot(x, x), max(dot(y, y), dot(z, z)));
		f32 scale = sqrt(m);
		float3 center = mul(this.mv, float4(sphere.xyz, 1.f)).xyz;
		return float4(center, sphere.w * scale);
	}

	f32 sphere_diameter_pixels(float4 sphere) {
		f32 d2 = dot(sphere.xyz, sphere.xyz);
		f32 r2 = sphere.w * sphere.w;
		return this.h * max(this.screen.x, this.screen.y) * sphere.w / sqrt(d2 - r2);
	}

	float4 transform_bounds(float4 lod_bounds, f32 error) {
		float4 bounds = this.transform_sphere(lod_bounds);
		// Place the error sphere at the closest point of the view-space lod bounds.
		f32 dist = max(bounds.z - bounds.w, 0.f);
		return float4(bounds.xy, dist, error);
	}

	bool is_imperceptible(float4 lod_bounds, f32 error) {
		float4 test = this.transform_bounds(lod_bounds, error);
		return this.sphere_diameter_pixels(test) < 1.f;
	}

	bool is_perceptible(float4 lod_bounds, f32 error) {
		float4 test = this.transform_bounds(lod_bounds, error);
		return this.sphere_diameter_pixels(test) >= 1.f;
	}
};

struct OccCull {
	float4x4 mvp;
	float2 screen;
	Tex2D hzb;
	Sampler hzb_sampler;

	static OccCull init(float4x4 mvp, uint2 res, Tex2D hzb, Sampler hzb_sampler) {
		OccCull ret;
		ret.mvp = mvp;
		ret.screen = float2(res);
		ret.hzb = hzb;
		ret.hzb_sampler = hzb_sampler;
		return ret;
	}

	bool cull(Aabb aabb) {
		float2 mi = float2(1.f, 1.f);
		float2 ma = float2(0.f, 0.f);
		f32 depth = 0.f;
		[unroll]
		for (u32 i = 0; i < 8; i++) {
            float4 clip = mul(this.mvp, float4(aabb.get_corner(i), 1.f));
            if (clip.w < 0.f) return false;
			float3 view = clip.xyz / clip.w;
			float2 uv = float2(view.x, -view.y) * 0.5f + 0.5f;
			mi = min(mi, uv);
			ma = max(ma, uv);
			depth = max(depth, view.z);
		}

		f32 width = (ma.x - mi.x) * this.screen.x * 0.5f;
		f32 height = (ma.y - mi.y) * this.screen.y * 0.5f;
		f32 level = ceil(log2(max(width, height)));
		f32 curr_depth = this.hzb.sample_mip(this.hzb_sampler, (mi + ma) * 0.5f, level).x;
		return depth < curr_depth;
		// return false;
	}
};
