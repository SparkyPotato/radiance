module cull;

import graph;
import passes.asset;

public struct BvhNodePointer {
	public u32 instance;
	public u32 node;
}

struct BvhQueueData {
	u32 len;
	u32x3 dispatch;
	BvhNodePointer pointers[];
}

public struct BvhQueue {
	BvhQueueData* data;

	public u32 len() {
		return this.data->len;
	}

	public BvhNodePointer get(u32 id) {
		return this.data->pointers[id];
	}

	public void push(BvhNodePointer pointer) {
		let id = atomic_add(this.data->len, 1);
		this.data->pointers[id] = pointer;
		if ((id & 63) == 0)
			atomic_add(this.data.dispatch.x, 1);
	}

	public void push(BvhNodePointer pointer, u32 count, u32 size) {
		let base = atomic_add(this.data->len, count);
		for (int i = 0; i < count; i++) {
			this.data->pointers[base + i] = pointer;
			pointer.node += size;
		}
		let next_mul = ((base + 63) / 64) * 64;
		if (next_mul - base < count)
			atomic_add(this.data.dispatch.x, 1);
	}
}

public struct MeshletPointer {
	public u32 instance;
	public u32 meshlet_offset;
}

struct MeshletQueueData {
	u32x3 dispatch;
	MeshletPointer pointers[];
}

public struct MeshletQueue {
	MeshletQueueData* data;

	public u32 len() {
		return this.data->dispatch.x;
	}

	public MeshletPointer get(u32 id) {
		return this.data->pointers[id];
	}

	public void push(MeshletPointer pointer) {
		let id = atomic_add(this.data.dispatch.x, 1u);
		this.data.pointers[id] = pointer;
	}
}

f32x4 normalize_plane(f32x4 p) {
	return p / length(p.xyz);
}

// https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
// https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
public struct Cull {
	f32x4x4 mv;
	f32x2 screen;
	f32 h;
	f32 near;
	f32x4 planes[5];
	u32x3 flips[5];

	public __init(f32x4x4 mv, f32x4x4 mvp, u32x2 res, f32 near, f32 h) {
		Cull ret;
		ret.mv = mv;
		ret.screen = f32x2(res);
		ret.h = h;
		ret.near = near;
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

	public bool frustum_cull(Aabb aabb) {
		[unroll]
		for (int i = 0; i < 5; i++) {
			let plane = this.planes[i];
			let sign_flipped = asfloat(asuint(aabb.half_extent) ^ this.flips[i]);
			if (dot(aabb.center + sign_flipped, plane.xyz) > -plane.w)
				return false;
		}
		return true;
	}

	f32 max_scale() {
		let x = this.mv._m00_m10_m20;
		let y = this.mv._m01_m11_m21;
		let z = this.mv._m02_m12_m22;
		let m = max(dot(x, x), max(dot(y, y), dot(z, z)));
		return sqrt(m);
	}

	f32 min_scale() {
		let x = this.mv._m00_m10_m20;
		let y = this.mv._m01_m11_m21;
		let z = this.mv._m02_m12_m22;
		let m = min(dot(x, x), min(dot(y, y), dot(z, z)));
		return sqrt(m);
	}

	f32 screenspace_scale(f32x4 bounds) {
		let center = mul(this.mv, f32x4(bounds.xyz, 1.f)).xyz;
		let radius = bounds.w * this.max_scale();

		let tdist2 = dot(center, center);
		let x2 = tdist2 - center.z * center.z;
		let x = sqrt(max(0.f, x2));
		let dist2 = tdist2 - radius * radius;
		let dist = sqrt(max(0.f, dist2));
		var t = (-radius * x + dist * center.z) / tdist2;

		let h = this.near - center.z;
		if (dist < 0.f || t * dist < this.near) {
			let tx = x + sqrt(radius * radius - h * h);
			t = this.near * rsqrt(tx * tx + this.near * this.near);
		}

		if (center.z + radius > this.near)
			return max(center.z - radius, this.near) * t;
		else
			return 0.f;
	}

	public bool is_imperceptible(f32x4 lod_bounds, f32 error) {
		let scale = this.screenspace_scale(lod_bounds);
		let thresh = this.min_scale() * error * max(this.screen.x, this.screen.y);
		return scale > thresh;
	}

	public bool is_perceptible(f32x4 lod_bounds, f32 error) {
		let scale = this.screenspace_scale(lod_bounds);
		let thresh = this.min_scale() * error * max(this.screen.x, this.screen.y);
		return scale <= thresh;
	}
}

vector<T, N> min8<T: __BuiltinFloatingPointType, let N : int>(
	vector<T, N> p0, 
	vector<T, N> p1, 
	vector<T, N> p2, 
	vector<T, N> p3, 
	vector<T, N> p4, 
	vector<T, N> p5, 
	vector<T, N> p6, 
	vector<T, N> p7
) {
	return min(p0, min(p1, min(p2, min(p3, min(p4, min(p5, min(p6, p7)))))));
}

vector<T, N> max8<T: __BuiltinFloatingPointType, let N : int>(
	vector<T, N> p0, 
	vector<T, N> p1, 
	vector<T, N> p2, 
	vector<T, N> p3, 
	vector<T, N> p4, 
	vector<T, N> p5, 
	vector<T, N> p6, 
	vector<T, N> p7
) {
	return max(p0, max(p1, max(p2, max(p3, max(p4, max(p5, max(p6, p7)))))));
}

public struct OccCull {
	f32x4x4 mvp;
	f32x2 screen;
	f32 near;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;

	public __init(f32x4x4 mvp, u32x2 res, f32 near, Tex<f32, D2> hzb, Sampler hzb_sampler) {
		OccCull ret;
		ret.mvp = mvp;
		ret.screen = f32x2(res);
		ret.near = near;
		ret.hzb = hzb;
		ret.hzb_sampler = hzb_sampler;
		return ret;
	}

	public bool cull(Aabb aabb) {
		let extent = aabb.half_extent * 2.f;
		let sx = mul(this.mvp, f32x4(extent.x, 0.f, 0.f, 0.f));
		let sy = mul(this.mvp, f32x4(0.f, extent.y, 0.f, 0.f));
		let sz = mul(this.mvp, f32x4(0.f, 0.f, extent.z, 0.f));

		let p0 = mul(this.mvp, f32x4(aabb.center - aabb.half_extent, 1.f));
		let p1 = p0 + sz;
		let p2 = p0 + sy;
		let p3 = p2 + sz;
		let p4 = p0 + sx;
		let p5 = p4 + sz;
		let p6 = p4 + sy;
		let p7 = p6 + sz;

		let depth = min8(p0, p1, p2, p3, p4, p5, p6, p7).w;
		if (depth < this.near)
			return false;

		let dp0 = p0.xy / p0.w;
		let dp1 = p1.xy / p1.w;
		let dp2 = p2.xy / p2.w;
		let dp3 = p3.xy / p3.w;
		let dp4 = p4.xy / p4.w;
		let dp5 = p5.xy / p5.w;
		let dp6 = p6.xy / p6.w;
		let dp7 = p7.xy / p7.w;
		var vaabb = f32x4(min8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7), max8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7));
		vaabb = vaabb.xwzy * f32x4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;

		let width = (vaabb.z - vaabb.x) * this.screen.x * 0.5f;
		let height = (vaabb.w - vaabb.y) * this.screen.y * 0.5f;
		let level = ceil(log2(max(width, height)));
		let curr_depth = this.hzb.sample_mip(this.hzb_sampler, (vaabb.xy + vaabb.zw) * 0.5f, level).x;
		return (this.near / depth) <= curr_depth;
	}
}

