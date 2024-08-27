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
	uint3 flips[5];

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
			ret.flips[i] = asuint(ret.planes[i].xyz) & 0x80000000;
		}
		return ret;
	} 

	bool frustum_cull(Aabb aabb) {
		[unroll]
		for (int i = 0; i < 5; i++) {
			float4 plane = this.planes[i];
			float3 sign_flipped = asfloat(asuint(aabb.half_extent) ^ this.flips[i]);
			if (dot(aabb.center + sign_flipped, plane.xyz) > -plane.w) return false;
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

template<typename T>
T min8(T p0, T p1, T p2, T p3, T p4, T p5, T p6, T p7) {
	return min(p0, min(p1, min(p2, min(p3, min(p4, min(p5, min(p6, p7)))))));
}

template<typename T>
T max8(T p0, T p1, T p2, T p3, T p4, T p5, T p6, T p7) {
	return max(p0, max(p1, max(p2, max(p3, max(p4, max(p5, max(p6, p7)))))));
}

struct OccCull {
	float4x4 mvp;
	float2 screen;
	f32 near;
	Tex2D hzb;
	Sampler hzb_sampler;

	static OccCull init(float4x4 mvp, uint2 res, f32 near, Tex2D hzb, Sampler hzb_sampler) {
		OccCull ret;
		ret.mvp = mvp;
		ret.screen = float2(res);
		ret.near = near;
		ret.hzb = hzb;
		ret.hzb_sampler = hzb_sampler;
		return ret;
	}

	bool cull(Aabb aabb) {
		float3 extent = aabb.half_extent * 2.f;
		float4 sx = mul(this.mvp, float4(extent.x, 0.f, 0.f, 0.f));
		float4 sy = mul(this.mvp, float4(0.f, extent.y, 0.f, 0.f));
		float4 sz = mul(this.mvp, float4(0.f, 0.f, extent.z, 0.f));

		float4 p0 = mul(this.mvp, float4(aabb.center - aabb.half_extent, 1.f));
		float4 p1 = p0 + sz;
		float4 p2 = p0 + sy;
		float4 p3 = p2 + sz;
		float4 p4 = p0 + sx;
		float4 p5 = p4 + sz;
		float4 p6 = p4 + sy;
		float4 p7 = p6 + sz;

		f32 depth = min8(p0, p1, p2, p3, p4, p5, p6, p7).w;
		if (depth < this.near) return false;

		float2 dp0 = p0.xy / p0.w;
		float2 dp1 = p1.xy / p1.w;
		float2 dp2 = p2.xy / p2.w;
		float2 dp3 = p3.xy / p3.w;
		float2 dp4 = p4.xy / p4.w;
		float2 dp5 = p5.xy / p5.w;
		float2 dp6 = p6.xy / p6.w;
		float2 dp7 = p7.xy / p7.w;
		float4 vaabb = float4(
			min8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7),
			max8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
		);
		vaabb = vaabb.xwzy * float4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;
		
		f32 width = (vaabb.z - vaabb.x) * this.screen.x * 0.5f;
		f32 height = (vaabb.w - vaabb.y) * this.screen.y * 0.5f;
		f32 level = ceil(log2(max(width, height)));
		f32 curr_depth = this.hzb.sample_mip(this.hzb_sampler, (vaabb.xy + vaabb.zw) * 0.5f, level).x;
		return (this.near / depth) <= curr_depth;
	}
};
