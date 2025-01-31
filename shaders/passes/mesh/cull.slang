module cull;

import graph;
import asset;

public struct NodePointer {
	public u32 instance;
	public u32 node_offset;
}

public struct PassStats {
	public u32 instances;
	public u32 candidate_meshlets;
	public u32 hw_meshlets;
	public u32 sw_meshlets;
}

public struct CullStats {
	public PassStats early;
	public PassStats late;
}

public PassStats* get_stats(CullStats* stats) {
	return EARLY ? &stats->early : &stats->late;
}

struct Dispatch {
	u32 count;
	u32x3 dispatch;
}

struct BvhQueueData {
	u32 len;
	Dispatch front;
	Dispatch back;
	NodePointer pointers[];
}

public struct BvhQueue {
	BvhQueueData* data;

	public u32 front_count() {
		return this.data->front.count;
	}

	public u32 back_count() {
		return this.data->back.count;
	}

	public NodePointer get_front(u32 id) {
		return this.data->pointers[id];
	}

	public NodePointer get_back(u32 id) {
		return this.data->pointers[this.data->len - id - 1];
	}

	public void push_front(NodePointer pointer) {
		let pos = wave_atomic_inc(this.data->front.count);
		let req = (pos + 1 + 7) >> 3;
		if ((pos & 7) == 0)
			wave_atomic_max(this.data->front.dispatch.x, req);
		this.data->pointers[pos] = pointer;
	}

	public void push_back(NodePointer pointer) {
		let pos = wave_atomic_inc(this.data->back.count);
		let req = (pos + 1 + 7) >> 3;
		if ((pos & 7) == 0)
			wave_atomic_max(this.data->back.dispatch.x, req);
		this.data->pointers[this.data->len - pos - 1] = pointer;
	}

	public void done_front() {
		done(&this.data->front);
	}

	public void done_back() {
		done(&this.data->back);
	}

	static void done(Dispatch* dispatch) {
		let prev = atomic_sub(dispatch->dispatch.x, 1);
		if (prev == 1)
			dispatch->count = 0;
	}
}

public struct CandidateMeshletQueue {
	BvhQueueData* data;

	public u32 count() {
		return EARLY ? this.data->front.count : this.data->back.count;
	}

	public NodePointer get(u32 id) {
		return EARLY ? this.data->pointers[id] : this.data->pointers[this.data->len - id - 1];
	}

	public void push_early(NodePointer pointer, u32 count) {
		let base = wave_atomic_add(this.data->front.count, count);
		let req = (base + count + 63) >> 6;
		wave_atomic_max(this.data->front.dispatch.x, req);
		let end = base + count;
		for (int i = base; i < end; i++) {
			this.data->pointers[i] = pointer;
			pointer.node_offset += sizeof(Meshlet);
		}
	}

	public void push_late(NodePointer pointer, u32 count) {
		let base = wave_atomic_add(this.data->back.count, count);
		let req = (base + count + 63) >> 6;
		wave_atomic_max(this.data->back.dispatch.x, req);
		let start = this.data->len - base - count;
		let end = this.data->len - base;
		for (int i = start; i < end; i++) {
			this.data->pointers[i] = pointer;
			pointer.node_offset += sizeof(Meshlet);
		}
	}
}

struct MeshletQueueData {
	u32 len;
	u32 early_hw;
	u32x3 hw;
	u32 early_sw;
	u32x3 sw;
	NodePointer pointers[];
}

public struct MeshletQueue {
	MeshletQueueData* data;

	public NodePointer get(u32 id) {
		return this.data->pointers[id];
	}

	public u32 hw_count() {
		return this.data->hw.x;
	}

	public u32 sw_count() {
		return this.data->sw.x;
	}

	public u32 hw_id(u32 id) {
		return (EARLY ? 0 : this.data->early_hw) + id;
	}

	public u32 sw_id(u32 id) {
		return this.data->len - (EARLY ? 0 : this.data->early_sw) - id - 1;
	}

	public NodePointer get_hw(u32 id) {
		return this.data->pointers[this.hw_id(id)];
	}

	public NodePointer get_sw(u32 id) {
		return this.data->pointers[this.sw_id(id)];
	}

	public void push_hw(NodePointer pointer) {
		if (EARLY)
			wave_atomic_inc(this.data->early_hw);
		let pos = wave_atomic_inc(this.data->hw.x);
		this.data->pointers[this.hw_id(pos)] = pointer;
	}

	public void push_sw(NodePointer pointer) {
		if (EARLY)
			wave_atomic_inc(this.data->early_sw);
		let pos = wave_atomic_inc(this.data->sw.x);
		this.data->pointers[this.sw_id(pos)] = pointer;
	}
}

f32x4 normalize_plane(f32x4 p) {
	return p / length(p.xyz);
}

public extern static const bool EARLY = false;

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

public bool occ_cull_aabb(ScreenAabb aabb, f32x2 screen, Tex2D<f32> hzb, Sampler hzb_sampler) {
	let hzb_size = 1 << firstbithigh(u32x2(screen) - 1);

	let min_texel = u32x2(max(aabb.min.xy, 0.f));
	let max_texel = u32x2(min(aabb.max.xy, screen - 1.f));
	let size = max_texel - min_texel + 1;
	let max_size = max(size.x, size.y);

	var mip = firstbithigh(max_size - 1) - 1;
	let smin = min_texel >> mip;
	let smax = max_texel >> mip;
	if (any(smax - smin > 1))
		mip += 1;

	let uv = ((f32x2(min_texel) + f32x2(max_texel)) * 0.5f) / hzb_size;
	let curr_depth = hzb.sample_mip(hzb_sampler, uv, mip);
	return aabb.max.z <= curr_depth;
}

public Camera occ_camera(Camera* cam) {
	if (EARLY)
		return cam[1];
	else
		return cam[0];
}

public Transform occ_transform(Instance instance, u64 frame) {
	if (EARLY)
		return instance.prev_transform(frame);
	else
		return instance.transform;
}

public struct Cull {
	f32 scale;
	f32 min_scale;
	f32x4x4 curr_mv;
	f32x4x4 curr_mvp;
	f32x4x4 prev_mvp;
	f32x2 screen;
	f32 h;
	f32 near;
	Tex2D<f32> hzb;
	Sampler hzb_sampler;

	public __init(Camera* camera, Instance instance, u64 frame, u32x2 res, Tex2D<f32> hzb, Sampler hzb_sampler) {
		Cull ret;

		let transform = instance.transform;
		let trans_mat = transform.mat();
		ret.scale = max3(transform.scale.x, transform.scale.y, transform.scale.z);
		ret.min_scale = min3(transform.scale.x, transform.scale.y, transform.scale.z);

		ret.curr_mv = mul(camera[0].view(), trans_mat);
		ret.curr_mvp = mul(camera[0].view_proj(), trans_mat);
		ret.prev_mvp = mul(occ_camera(camera).view_proj(), occ_transform(instance, frame).mat());

		ret.screen = f32x2(res);
		ret.h = camera[0].h;
		ret.near = camera[0].near;

		ret.hzb = hzb;
		ret.hzb_sampler = hzb_sampler;

		return ret;
	}

	// https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
	// https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
	public bool in_frustum(Aabb aabb) {
		f32x4 planes[] = { normalize_plane(this.curr_mvp[3] + this.curr_mvp[0]),
						   normalize_plane(this.curr_mvp[3] - this.curr_mvp[0]),
						   normalize_plane(this.curr_mvp[3] + this.curr_mvp[1]),
						   normalize_plane(this.curr_mvp[3] - this.curr_mvp[1]), normalize_plane(this.curr_mvp[2]) };
		for (int i = 0; i < planes.getCount(); i++) {
			let flip = asuint(planes[i].xyz) & 0x80000000;
			let sign_flipped = asfloat(asuint(aabb.half_extent) ^ flip);
			if (dot(aabb.center + sign_flipped, planes[i].xyz) <= -planes[i].w)
				return false;
		}
		return true;
	}

	// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere (Michael Mara, Morgan McGuire).
	// We get the projected bounds on the axis that is the longest upon projection (need to be conservative!),
	// which is the one from (0, 0) to the sphere's center.
	public f32 error_distance(f32x4 bounds) {
		let center = mul(this.curr_mv, f32x4(bounds.xyz, 1.f)).xyz;
		let radius = bounds.w * this.scale;

		if (center.y + radius <= this.near)
			return 0.f;

		let dist2 = dot(center, center);
		let a = sqrt(dist2 - center.y * center.y);
		let proj_center = f32x2(a, center.y);
		let t2 = dist2 - radius * radius;
		var t = sqrt(max(t2, 0.f));

		// cos(theta) = t / dist
		// sin(theta) = r / dist
		// T = t * rotate(theta) * proj_center / dist,
		// removing the dist divide in cos, sin
		// ncos(theta) = t
		// nsin(theta) = r
		// rotate(theta) == rotate(ntheta) / dist
		// therefore, T = t * rotate(ntheta) * proj_center / dist2
		// saving us two divides and a sqrt!
		let ncos = t;
		let nsin = radius;
		var wt_z = dot(f32x2(-nsin, ncos), proj_center) / dist2;
		let t_z = t * wt_z;

		// If intersecting the near plane, funky business is required.
		if (t2 < 0.f || t_z < this.near) {
			let off = this.near - center.y;
			let k = sqrt(radius * radius - off * off);
			let t = f32x2(a + k, this.near);
			wt_z = t.y / length(t);
		}

		// Project the sphere to an equivalently sized sphere along the z axis,
		// so error_scale can be used to monotonically compare.
		return wt_z * max(center.y - radius, this.near);
	}

	public f32 error_scale() {
		return sin(atan(this.h / this.screen.y));
		// Don't divide by 2 because the error is already divided by 2 during build.
		return this.screen.y * this.h;
	}

	bool should_visit_nv(f32x4 lod_bounds, f32 parent_error) {
		let sphere_dist = length(mul(this.curr_mv, f32x4(lod_bounds.xyz, 1.f)).xyz);
		let err_dist = max(this.near, sphere_dist - lod_bounds.w * this.scale);
		let err_over_dist = parent_error * this.scale / err_dist;
		return err_over_dist >= this.error_scale();
	}

	public bool should_visit_bvh(f32x4 lod_bounds, f32 parent_error) {
		return this.should_visit_nv(lod_bounds, parent_error);

		let min_dist = this.error_scale() * this.min_scale * parent_error;
		return this.error_distance(lod_bounds) <= min_dist;
	}

	public bool2 should_render(f32x4 lod_bounds, f32 error, f32 edge) {
		return bool2(!this.should_visit_nv(lod_bounds, error), true);

		let dist = this.error_distance(lod_bounds);
		let scale = this.error_scale();
		let min_dist = scale * this.min_scale * error;
		let edge_scale = (scale / 32.f) * this.scale * edge;
		return bool2(dist > min_dist, /*dist < edge_scale*/ true);
	}

	public bool unoccluded(Aabb aabb) {
		if (let saabb = project_aabb(this.prev_mvp, this.near, aabb)) {
			var uaabb = saabb;
			let scale = this.screen * 0.5f;
			uaabb.min.xy = uaabb.min.xy * scale;
			uaabb.max.xy = uaabb.max.xy * scale;
			return !occ_cull_aabb(uaabb, this.screen, this.hzb, this.hzb_sampler);
		}
		return true;
	}
}

