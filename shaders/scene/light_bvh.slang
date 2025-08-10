module light_bvh;

import graph;
import asset;

// https://gist.github.com/natevm/6618402427ad6466bf555d67602adfa8
// With several fixes and modified for light trees.

struct LightBvhNode {
	f32x3 aabb_min;
	// LSB is if is inner node.
	u32 L;
	f32x3 aabb_max;
	// LSB is if is inner node.
	u32 R;

	__init(u32 left, u32 right, f32x2x3 bounds) {
		aabb_min = bounds[0];
		aabb_max = bounds[1];
		L = left;
		R = right;
	}
}

struct PushConstants {
	LightBvhNode* bvh_nodes;
	u32* atomic;
	u32* cluster_indices;
	u64* codes;
	u32* parent_ids;
	Vertex* vertices;
	u32x3* indices;
	f32x3 root_bounds[2];
	u32 tri_count;
}

[vk::push_constant]
PushConstants Constants;

static const u32 INVALID_ID = -1;

[shader("compute")]
[numthreads(128, 1, 1)]
void hploc_sfc(u32x3 tid: SV_DispatchThreadID) {
	let id = tid.x;
	if (id >= Constants.tri_count)
		return;
	if (id == 0)
		*Constants.atomic = 0;

	var center = load_triangle_center(id);
	center = (center - Constants.root_bounds[0]) / (Constants.root_bounds[1] - Constants.root_bounds[0]);

	Constants.codes[id] = morton64_encode(center);
	Constants.cluster_indices[id] = id << 1;
	Constants.parent_ids[id] = INVALID_ID;
}

// Force wave size to 32
[shader("compute")]
[numthreads(32, 1, 1)]
void hploc_build(u32x3 tid: SV_DispatchThreadID) {
	var i = tid.x;
	var N = Constants.tri_count;
	var L = i;
	var R = i;
	var lane_active = i < N;

	while (WaveActiveAnyTrue(lane_active)) {
		var split = INVALID_ID;
		if (lane_active) {
			u32 previous_id;
			if (find_parent_id(L, R, N) == R) {
				previous_id = atomic_exchange(Constants.parent_ids[R], L);
				if (previous_id != INVALID_ID) {
					split = R + 1;
					R = previous_id;
				}
			} else {
				previous_id = atomic_exchange(Constants.parent_ids[L - 1], R);
				if (previous_id != INVALID_ID) {
					split = L;
					L = previous_id;
				}
			}
			if (previous_id == INVALID_ID)
				lane_active = false;
		}

		let size = R - L + 1;
		let final = lane_active && size == N;
		var wave_mask = WaveActiveBallot((lane_active && (size > 16)) || final).x;
		while (wave_mask != 0) {
			let lane_id = firstbitlow(wave_mask);
			ploc_merge(lane_id, L, R, split, final);
			wave_mask = wave_mask & (wave_mask - 1);  // This unsets the `lane_id` bit.
		}
	}
}

void ploc_merge(u32 lane_id, u32 L, u32 R, u32 split, bool final) {
	let wid = WaveGetLaneIndex();

	let l_start = WaveShuffle(L, lane_id);
	let l_end = WaveShuffle(split, lane_id);
	let r_start = l_end;
	let r_end = WaveShuffle(R, lane_id) + 1;

	var cluster_index = INVALID_ID;
	let num_left = load_cluster_indices(l_start, l_end, cluster_index, 0);
	let num_right = load_cluster_indices(r_start, r_end, cluster_index, num_left);
	var num_prims = num_left + num_right;

	f32x2x3 bounds = f32x2x3(f32x3(1.0 / 0.0), f32x3(-1.0 / 0.0));
	if (cluster_index != INVALID_ID)
		bounds = load_aabb(cluster_index);

	let threshold = WaveShuffle(final, lane_id) ? 1 : 16;
	while (num_prims > threshold) {
		num_prims = merge_clusters_to_create_bvh_node(num_prims, cluster_index, bounds);
	}
	store_cluster_indices(num_left + num_right, cluster_index, l_start);
}

u32 merge_clusters_to_create_bvh_node(u32 num_prims, inout u32 cluster_index, inout f32x2x3 bounds) {
	let wid = WaveGetLaneIndex();

	let nn = find_nearest_neighbor(num_prims, bounds);
	let n_i = decode_relative_offset(wid, nn & PAYLOAD_MASK);
	let n_i_n_i = decode_relative_offset(n_i, WaveShuffle(nn, n_i) & PAYLOAD_MASK);
	let symmetric_match_found = wid == n_i_n_i;
	let lane_is_left_neighbor = wid < n_i;

	let left_cluster = cluster_index;
	let right_cluster = WaveShuffle(cluster_index, n_i);
	let left_bounds = bounds;
	f32x2x3 right_bounds;
	right_bounds[0] = WaveShuffle(bounds[0], n_i);
	right_bounds[1] = WaveShuffle(bounds[1], n_i);

	let lane_has_cluster = wid < num_prims;
	let lane_is_creating = lane_has_cluster && symmetric_match_found && lane_is_left_neighbor;
	let append_count = WaveActiveCountBits(lane_is_creating);
	var allocated_count = INVALID_ID;
	if (WaveIsFirstLane())
		allocated_count = atomic_add(*Constants.atomic, append_count);
	allocated_count = WaveReadLaneFirst(allocated_count);

	let base_node_offset = (Constants.tri_count - 1) - (append_count + allocated_count);
	let bvh_index_prefix = WavePrefixCountBits(lane_is_creating);

	var new_cluster_id = lane_has_cluster ? cluster_index : INVALID_ID;
	// If we're part of a merge
	if (lane_has_cluster && symmetric_match_found) {
		if (lane_is_creating) {
			let bvh_index = base_node_offset + bvh_index_prefix;
			bounds = f32x2x3(min(left_bounds[0], right_bounds[0]), max(left_bounds[1], right_bounds[1]));
			Constants.bvh_nodes[bvh_index] = LightBvhNode(left_cluster, right_cluster, bounds);
			new_cluster_id = (bvh_index << 1) | 1;
		} else {
			new_cluster_id = INVALID_ID;
		}
	}

	let ps = new_cluster_id != INVALID_ID;
	let valid_lanes = WaveActiveBallot(ps).x;
	let compacted_id = find_nth_bit(valid_lanes, wid);
	bounds[0] = WaveShuffle(bounds[0], compacted_id);
	bounds[1] = WaveShuffle(bounds[1], compacted_id);
	cluster_index = WaveShuffle(new_cluster_id, compacted_id);
	cluster_index = compacted_id != INVALID_ID ? cluster_index : INVALID_ID;
	return countbits(valid_lanes);
}

groupshared u32 CachedNeighbors[32];
static const u32 SEARCH_RADIUS_BITS = 3;
static const u32 SEARCH_RADIUS = 1 << SEARCH_RADIUS_BITS;
static const u32 PAYLOAD_MASK = (1 << (SEARCH_RADIUS_BITS + 1)) - 1;

u32 find_nearest_neighbor(u32 prim_count, f32x2x3 bounds) {
	u32 wid = WaveGetLaneIndex();
	CachedNeighbors[wid] = INVALID_ID;
	workgroup_wave_sync_barrier();

	var min_area_index = u32(-1);
	for (u32 r = 1; r <= SEARCH_RADIUS && (wid + r < prim_count); r++) {
		let neighbor = wid + r;
		f32x2x3 neighbor_bounds;
		neighbor_bounds[0] = WaveShuffle(bounds[0], neighbor);
		neighbor_bounds[1] = WaveShuffle(bounds[1], neighbor);
		let new_area = distance_factor(bounds, neighbor_bounds);
		// Gives us SEARCH_RADIUS_BITS + 1 bits to store our payload
		let new_area_i = (asuint(new_area) << 1) & ~PAYLOAD_MASK;
		let us_to_neighbor = new_area_i | encode_relative_offset(wid, neighbor);
		let neighbor_to_us = new_area_i | encode_relative_offset(neighbor, wid);
		min_area_index = min(min_area_index, us_to_neighbor);
		atomic_min(CachedNeighbors[neighbor], neighbor_to_us, Scope.Workgroup);
	}
	atomic_min(CachedNeighbors[wid], min_area_index, Scope.Workgroup);
	workgroup_wave_sync_barrier();

	return CachedNeighbors[wid];
}

u32 encode_relative_offset(i32 wid, i32 neighbor) {
	let s_offset = neighbor - wid;
	let u_offset = abs(s_offset) - 1;
	return (u_offset << 1) | (u32(s_offset) >> 31);
}

i32 decode_relative_offset(i32 wid, u32 offset) {
	let off = i32((offset >> 1) + 1);
	return wid + ((offset & 1) == 0 ? off : -off);
}

u32 load_cluster_indices(u32 start, u32 end, inout u32 cluster_index, u32 offset) {
	let wid = WaveGetLaneIndex();
	var num_indices = min(end - start, 16);
	let index_id = i32(wid - offset);
	let lane_active = index_id >= 0 && index_id < num_indices;
	if (lane_active)
		cluster_index = load(Constants.cluster_indices, start + index_id, Scope.Queue);
	let num_valid = WaveActiveCountBits(lane_active);
	num_indices = min(num_indices, num_valid);
	return num_indices;
}

void store_cluster_indices(u32 orig_num_prims, u32 cluster_index, u32 l_start) {
	let wid = WaveGetLaneIndex();
	if (wid < orig_num_prims) {
		store(Constants.cluster_indices, l_start + wid, cluster_index, Scope.Queue);
	}
}

u64 separate_bits_64(u64 n) {
	n &= 0b0000000000000000000000000000000000000000001111111111111111111111ull;
	n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
	n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
	n = (n ^ (n << 8)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
	n = (n ^ (n << 4)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
	n = (n ^ (n << 2)) & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
	return n;
};

u64 morton64_encode(f32x3 xyz) {
	let ixyz = u64x3(xyz * f32(1 << 20));
	return separate_bits_64(ixyz.x) | (separate_bits_64(ixyz.y) << 1) | (separate_bits_64(ixyz.z) << 2);
}

f32 surface_area(f32x2x3 bounds) {
	let d = bounds[1] - bounds[0];
	return max(2.f * (d.x * d.y + d.x * d.z + d.y * d.z), 0.f);
}

f32 distance_factor(f32x2x3 a, f32x2x3 b) {
	if (any(a[1] < a[0]) || any(b[1] < b[0]))
		return 1e38f;
	return surface_area(f32x2x3(max(a[1], b[1]), min(a[0], b[0])));
}

u64 delta(i32 a, i32 b, u32 n) {
	if (a < 0 || b >= n)
		return -1;
	let c = Constants.codes[a] ^ Constants.codes[b];
	if (c == 0)
		return a ^ (a + 1);
	return c;
}

u32 find_parent_id(i32 a, i32 b, u32 n) {
	if (a == 0 || (b != n && (delta(b, b + 1, n) < delta(a - 1, a, n))))
		return b;
	else
		return a - 1;
}

u32 find_nth_bit(u32 mask, u32 n) {
	var count = 0;
	for (u32 i = 0; i < 32; i++) {
		if ((mask & (1 << i)) != 0) {
			if (count == n)
				return i;
			count++;
		}
	}
	return INVALID_ID;
}

void load_triangle(u32 prim, out f32x3 v[3]) {
	let tri = Constants.indices[prim];
	v[0] = Constants.vertices[tri.x].position;
	v[1] = Constants.vertices[tri.y].position;
	v[2] = Constants.vertices[tri.z].position;
}

f32x2x3 load_triangle_bounds(u32 prim) {
	f32x3 v[3];
	load_triangle(prim, v);
	return f32x2x3(min3(v[0], v[1], v[2]), max3(v[0], v[1], v[2]));
}

f32x2x3 load_bvh_bounds(u32 node) {
	let node = Constants.bvh_nodes[node];
	return f32x2x3(node.aabb_min, node.aabb_max);
}

f32x2x3 load_aabb(u32 cluster) {
	let is_bvh_node = (cluster & 1) == 1;
	let index = cluster >> 1;
	if (is_bvh_node)
		return load_bvh_bounds(index);
	else
		return load_triangle_bounds(index);
}

f32x3 load_triangle_center(u32 prim) {
	f32x3 v[3];
	load_triangle(prim, v);
	return (v[0] + v[1] + v[2]) / 3.f;
}

f32x3 load_bvh_center(u32 node) {
	let bounds = load_bvh_bounds(node);
	return (bounds[0] + bounds[1]) * 0.5f;
}

f32x3 load_center(u32 cluster) {
	let is_bvh_node = (cluster & 1) == 1;
	let index = cluster >> 1;
	if (is_bvh_node)
		return load_bvh_center(index);
	else
		return load_triangle_center(index);
}
