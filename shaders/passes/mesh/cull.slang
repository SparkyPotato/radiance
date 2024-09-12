module cull;

import graph;
import asset;

public struct NodePointer {
	public u32 instance;
	public u32 node_offset;
}

struct BvhQueueData {
	u32 len;
	u32x3 dispatch;
	NodePointer pointers[];
}

public struct BvhQueue {
	BvhQueueData* data;

	public u32 len() {
		return this.data->len;
	}

	public NodePointer get(u32 id) {
		return this.data->pointers[id];
	}

	bool in_bounds(u32 len, u32 i) {
		return sizeof(u32x4) + (i + 1) * sizeof(NodePointer) < len;
	}

	u32 clamp_end(u32 len, u32 end) {
		return min((len - sizeof(u32x4)) / sizeof(NodePointer), end);
	}

	u32 reserve(u32 len, u32 bits, u32 count) {
		let base = atomic_add(this.data->len, count);
		let extra = select(base == 0, 1, 0);
		let curr = base >> bits;
		let req = this.clamp_end(len, base + count) >> bits;
		let add = extra + req - curr;
		if (add > 0)
			atomic_add(this.data->dispatch.x, add);
		return base;
	}

	public void push(NodePointer pointer) {
		this.push(-1, pointer);
	}

	public void push(u32 len, NodePointer pointer) {
		this.push_inner(len, 3, pointer);
	}

	public void push_meshlet(u32 len, NodePointer pointer) {
		this.push_inner(len, 6, pointer);
	}

	void push_inner(u32 len, u32 bits, NodePointer pointer) {
		let rel = WavePrefixCountBits(true);
		let total = WaveActiveCountBits(true);
		u32 pos;
		if (WaveIsFirstLane())
			pos = this.reserve(len, bits, total);
		pos = WaveReadLaneFirst(pos) + rel;
		if (this.clamp_end(len, pos + 1) != pos)
			this.data->pointers[pos] = pointer;
	}

	public void push_meshlets(u32 len, NodePointer pointer, u8 count) {
		let rel = WavePrefixSum(u32(count));
		let total = WaveActiveSum(u32(count));
		u32 base;
		if (WaveIsFirstLane())
			base = this.reserve(len, 6, total);
		base = WaveReadLaneFirst(base) + rel;
		let end = this.clamp_end(len, base + count);
		for (int i = base; i < end; i++) {
			this.data->pointers[i] = pointer;
			pointer.node_offset += sizeof(Meshlet);
		}
	}

	public void done() {
		let prev = atomic_sub(this.data->dispatch.x, 1);
		if (prev == 1)
			this.data->len = 0;
	}
}

struct MeshletQueueData {
	u32x3 dispatch;
	NodePointer pointers[];
}

public struct MeshletQueue {
	MeshletQueueData* data;

	public u32 len() {
		return this.data->dispatch.x;
	}

	public NodePointer get(u32 id) {
		return this.data->pointers[id];
	}

	public void push(u32 len, NodePointer pointer) {
		let us = WaveMatch(u64(this.data));
		let rel = WavePrefixCountBits(true);
		let total = WaveActiveCountBits(true);
		u32 pos;
		if (WaveIsFirstLane())
			pos = atomic_add(this.data->dispatch.x, total);
		pos = WaveReadLaneFirst(pos) + rel;
		if ((pos + 1) * sizeof(NodePointer) + sizeof(u32x3) < len)
			this.data.pointers[pos] = pointer;
	}
}

f32x4 normalize_plane(f32x4 p) {
	return p / length(p.xyz);
}

public extern static const bool EARLY = false;

Camera occ_camera(Camera* cam) {
	if (EARLY)
		return cam[1];
	else
		return cam[0];
}

vector<T, N> min8<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> p0, vector<T, N> p1, vector<T, N> p2,
															   vector<T, N> p3, vector<T, N> p4, vector<T, N> p5,
															   vector<T, N> p6, vector<T, N> p7) {
	return min(p0, min(p1, min(p2, min(p3, min(p4, min(p5, min(p6, p7)))))));
}

vector<T, N> max8<T : __BuiltinFloatingPointType, let N : int>(vector<T, N> p0, vector<T, N> p1, vector<T, N> p2,
															   vector<T, N> p3, vector<T, N> p4, vector<T, N> p5,
															   vector<T, N> p6, vector<T, N> p7) {
	return max(p0, max(p1, max(p2, max(p3, max(p4, max(p5, max(p6, p7)))))));
}

public struct ScreenAabb {
	public f32x3 min;
	public f32x3 max;
}

// https://zeux.io/2023/01/12/approximate-projected-bounds/
public Optional<ScreenAabb> project_aabb(f32x4x4 mvp, f32 near, Aabb aabb) {
	let extent = aabb.half_extent * 2.f;
	let sx = mul(mvp, f32x4(extent.x, 0.f, 0.f, 0.f));
	let sy = mul(mvp, f32x4(0.f, extent.y, 0.f, 0.f));
	let sz = mul(mvp, f32x4(0.f, 0.f, extent.z, 0.f));

	let p0 = mul(mvp, f32x4(aabb.center - aabb.half_extent, 1.f));
	let p1 = p0 + sz;
	let p2 = p0 + sy;
	let p3 = p2 + sz;
	let p4 = p0 + sx;
	let p5 = p4 + sz;
	let p6 = p4 + sy;
	let p7 = p6 + sz;

	let depth = min8(p0, p1, p2, p3, p4, p5, p6, p7).w;
	if (depth < near)
		return none;

	let dp0 = p0.xyz / p0.w;
	let dp1 = p1.xyz / p1.w;
	let dp2 = p2.xyz / p2.w;
	let dp3 = p3.xyz / p3.w;
	let dp4 = p4.xyz / p4.w;
	let dp5 = p5.xyz / p5.w;
	let dp6 = p6.xyz / p6.w;
	let dp7 = p7.xyz / p7.w;
	let min = min8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7);
	let max = max8(dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7);
	var vaabb = f32x4(min.xy, max.xy);
	vaabb = vaabb.xwzy * f32x4(0.5f, -0.5f, 0.5f, -0.5f) + 0.5f;
	ScreenAabb ret = { f32x3(vaabb.xy, min.z), f32x3(vaabb.zw, max.z) };
	return ret;
}

public struct Cull {
	f32 scale;
	f32 min_scale;
	f32x4x4 mv;
	f32x4x4 mvp;
	f32x2 screen;
	f32 h;
	f32 near;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	f32x4 planes[5];
	u32x3 flips[5];

	public __init(Camera* camera, Transform transform, u32x2 res, Tex<f32, D2> hzb, Sampler hzb_sampler) {
		Cull ret;
		let trans_mat = transform.mat();
		ret.scale = max3(transform.scale.x, transform.scale.y, transform.scale.z);
		ret.min_scale = min3(transform.scale.x, transform.scale.y, transform.scale.z);
		ret.mv = mul(camera[0].view, trans_mat);
		ret.mvp = mul(occ_camera(camera).view_proj, trans_mat);
		ret.screen = f32x2(res);
		ret.h = camera[0].h;
		ret.near = camera[0].near;
		ret.hzb = hzb;
		ret.hzb_sampler = hzb_sampler;
		let mvp = mul(camera[0].view_proj, trans_mat);
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

	// https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
	// https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
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

	f32 project_error(f32x4 bounds, f32 error) {
		let center = mul(this.mv, f32x4(bounds.xyz, 1.f)).xyz;
		let radius = bounds.w * this.scale;
		let depth = max(center.z - radius, this.near);
		let err = error * this.scale;
		let closest = f32x3(center.xy, depth);
		let d2 = dot(closest, closest);
		let r2 = err * err;
		return (this.h * err * max(this.screen.x, this.screen.y)) / sqrt(max(d2 - r2, 0.f));
	}

	public bool should_render(f32x4 lod_bounds, f32 error) {
		let err = this.project_error(lod_bounds, error);
		return err < 1.f;
	}

	public bool oc_cull(Aabb aabb) {
		if (let saabb = project_aabb(this.mvp, this.near, aabb)) {
			let size = (saabb.max - saabb.min).xy * this.screen * 0.5f;
			let level = ceil(log2(max(size.x, size.y)));
			let curr_depth = this.hzb.sample_mip(this.hzb_sampler, (saabb.min + saabb.max).xy * 0.5f, level);
			return saabb.min.z < curr_depth;
		}
		return false;
	}
}

