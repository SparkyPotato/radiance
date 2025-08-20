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
	public u32 overflow;
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

	public bool push_front(NodePointer pointer) {
		let pos = wave_atomic_inc(this.data->front.count);
		let other = atomic_load(this.data->back.count);
		if (pos + other >= this.data->len)
			return true;

		let req = (pos + 1 + 7) >> 3;
		if ((pos & 7) == 0)
			wave_atomic_max(this.data->front.dispatch.x, req);
		this.data->pointers[pos] = pointer;
		return false;
	}

	public bool push_back(NodePointer pointer) {
		let pos = wave_atomic_inc(this.data->back.count);
		let other = atomic_load(this.data->front.count);
		if (pos + other >= this.data->len)
			return true;

		let req = (pos + 1 + 7) >> 3;
		if ((pos & 7) == 0)
			wave_atomic_max(this.data->back.dispatch.x, req);
		this.data->pointers[this.data->len - pos - 1] = pointer;
		return false;
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

	public bool push_early(NodePointer pointer, u32 count) {
		let base = wave_atomic_add(this.data->front.count, count);
		let other = atomic_load(this.data->back.count);
		if (base + other + count >= this.data->len)
			return true;

		let req = (base + count + 63) >> 6;
		wave_atomic_max(this.data->front.dispatch.x, req);
		let end = base + count;
		for (int i = base; i < end; i++) {
			this.data->pointers[i] = pointer;
			pointer.node_offset += sizeof(Meshlet);
		}
		return false;
	}

	public bool push_late(NodePointer pointer, u32 count) {
		let base = wave_atomic_add(this.data->back.count, count);
		let other = atomic_load(this.data->front.count);
		if (base + other + count >= this.data->len)
			return true;

		let req = (base + count + 63) >> 6;
		wave_atomic_max(this.data->back.dispatch.x, req);
		let start = this.data->len - base - count;
		let end = this.data->len - base;
		for (int i = start; i < end; i++) {
			this.data->pointers[i] = pointer;
			pointer.node_offset += sizeof(Meshlet);
		}
		return false;
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

	public bool push_hw(NodePointer pointer) {
		if (EARLY)
			wave_atomic_inc(this.data->early_hw);
		let pos = wave_atomic_inc(this.data->hw.x);
		let id = this.hw_id(pos);
		let other = atomic_load(this.data->sw.x);
		if (id + other >= this.data->len)
			return true;

		this.data->pointers[id] = pointer;
		return false;
	}

	public bool push_sw(NodePointer pointer) {
		if (EARLY)
			wave_atomic_inc(this.data->early_sw);
		let pos = wave_atomic_inc(this.data->sw.x);
		let id = this.sw_id(pos);
		let other = atomic_load(this.data->hw.x);
		if (id + other >= this.data->len)
			return true;

		this.data->pointers[id] = pointer;
		return false;
	}
}

f32x4 normalize_plane(f32x4 p) {
	return p / length(p.xyz);
}

public extern static const bool EARLY = false;

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
	let max_texel = u32x2(min(aabb.max.xy, hzb_size - 1.f));
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

public Transform occ_transform(Instance* instance, u64 frame) {
	if (EARLY)
		return instance->prev_transform(frame);
	else
		return instance->transform;
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

	public __init(Camera* camera, Instance* instance, u64 frame, u32x2 res, Tex2D<f32> hzb, Sampler hzb_sampler) {
		Cull ret;

		let transform = instance->transform;
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

	// the metric I came up with was pretty bad, so steal nvidia's i guess:
	// https://github.com/nvpro-samples/vk_lod_clusters/blob/main/shaders/traversal_run.comp.glsl
	public f32 error_over_dist(f32x4 lod_bounds, f32 error) {
		let sphere_dist = length(mul(this.curr_mv, f32x4(lod_bounds.xyz, 1.f)).xyz);
		let err_dist = max(this.near, sphere_dist - lod_bounds.w * this.scale);
		return error * this.scale / err_dist;
	}

	public f32 threshold_for_pix(f32 pix) {
		return sin(atan(this.h * pix / this.screen.y));
	}

	public bool should_visit_bvh(f32x4 lod_bounds, f32 parent_error) {
		let err_over_dist = this.error_over_dist(lod_bounds, parent_error);
		let thresh = this.threshold_for_pix(1.f);
		return err_over_dist >= thresh;
	}

	public bool should_render(f32x4 lod_bounds, f32 error) {
		let err_over_dist = this.error_over_dist(lod_bounds, error);
		let thresh = this.threshold_for_pix(1.f);
		return err_over_dist < thresh;
	}

	public bool hw_or_sw(Aabb aabb, f32 edge) {
		let sphere = f32x4(aabb.center, length(aabb.half_extent));
		let err_over_dist = this.error_over_dist(sphere, edge);
		let thresh = this.threshold_for_pix(16.f);
		return err_over_dist >= thresh;
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
