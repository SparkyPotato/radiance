module light_tree_build;

import graph;
import graph.util.color;

import light_tree;
import light_tree_build_interface;

extern struct Interface : LightTreeBuildInterface;

// [H-PLOC: Hierarchical Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction]
// https://gist.github.com/natevm/6618402427ad6466bf555d67602adfa8
// With several fixes and modified for light trees.

struct PushConstants {
	Interface build;
	TempBvhNode* bvh_nodes;
	u32* atomic;
	u32* cluster_indices;
	u64* codes;
	u32* parent_ids;
	f32x3 root_bounds[2];
	u32 prim_count;
}

[vk::push_constant]
PushConstants Constants;

static const u32 INVALID_ID = -1;

[shader("compute")]
[numthreads(128, 1, 1)]
void hploc_sfc(u32x3 tid: SV_DispatchThreadID) {
	let id = tid.x;
	if (id >= Constants.prim_count)
		return;
	if (id == 0)
		*Constants.atomic = 0;

	var center = Constants.build.load_leaf_center(id);
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
	var N = Constants.prim_count;
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
		var wave_mask = WaveActiveBallot((lane_active && size > 16) || final).x;
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
	let num_clusters = load_cluster_indices(r_start, r_end, cluster_index, num_left);
	if (WaveIsFirstLane())
		printf("num_clusters: %d (%d, %d, %d)", num_clusters, l_start, l_end, r_end);

	NodeData data;
	data.aabb_min = f32x3(FLT_MAX);
	data.aabb_max = f32x3(-FLT_MAX);
	if (cluster_index != INVALID_ID) {
		data = load_node_data(cluster_index);
	}

	let threshold = WaveShuffle(final, lane_id) ? 1 : 16;
	var clusters_left = num_clusters;
	while (clusters_left > threshold) {
		clusters_left = merge_clusters_to_create_bvh_node(clusters_left, cluster_index, data);
		if (WaveIsFirstLane())
			printf("clusters_left: %d", clusters_left);
	}
	if (WaveIsFirstLane())
		printf("merged");
	store_cluster_indices(num_clusters, cluster_index, l_start);
}

u32 merge_clusters_to_create_bvh_node(u32 num_clusters, inout u32 cluster_index, inout NodeData data) {
	let wid = WaveGetLaneIndex();

	let nn = find_nearest_neighbor(num_clusters, data);
	let n_i = decode_relative_offset(wid, nn & PAYLOAD_MASK);
	let n_i_n_i = decode_relative_offset(n_i, WaveShuffle(nn, n_i) & PAYLOAD_MASK);
	let symmetric_match_found = wid == n_i_n_i;
	let lane_is_left_neighbor = wid < n_i;

	let left_cluster = cluster_index;
	let right_cluster = WaveShuffle(cluster_index, n_i);
	let left_data = data;
	let right_data = NodeData.wave_shuffle(data, n_i);

	let lane_has_cluster = wid < num_clusters;
	let other_lane_has_cluster = n_i < num_clusters;
	let lane_is_creating = lane_has_cluster && other_lane_has_cluster && symmetric_match_found && lane_is_left_neighbor;
	let append_count = WaveActiveCountBits(lane_is_creating);
	u32 allocated_count;
	if (WaveIsFirstLane())
		allocated_count = atomic_add(*Constants.atomic, append_count);
	allocated_count = WaveReadLaneFirst(allocated_count);

	if (lane_has_cluster)
		printf("%d = %d => %d, %d", wid, cluster_index, n_i, n_i_n_i);

	let base_node_offset = (Constants.prim_count - 1) - (append_count + allocated_count);
	let bvh_index_prefix = WavePrefixCountBits(lane_is_creating);

	var new_cluster_index = INVALID_ID;
	if (lane_has_cluster) {
		new_cluster_index = cluster_index;
		if (symmetric_match_found) {
			if (lane_is_creating) {
				let bvh_index = base_node_offset + bvh_index_prefix;
				data = NodeData(left_data, right_data);
				new_cluster_index = (bvh_index << 1) | 1;

				let bvh_node = TempBvhNode(data, left_cluster, right_cluster);
				let node = LightTreeNode(left_data, right_data, left_cluster, right_cluster);
				if (bvh_index < Constants.prim_count - 1) {
					store(Constants.bvh_nodes, bvh_index, bvh_node, Scope.Queue);
					Constants.build.store_node(bvh_index, node, data);
				}
			} else {
				new_cluster_index = INVALID_ID;
			}
		}
	}

	if (append_count + allocated_count > Constants.prim_count - 1) {
		if (WaveIsFirstLane())
			printf("overflow!");
		return 0;
	}

	CachedNeighbors[wid] = 0;
	workgroup_wave_sync_barrier();

	let is_lane_valid = new_cluster_index != INVALID_ID;
	let compacted_data_dst = WavePrefixCountBits(is_lane_valid);
	if (is_lane_valid)
		CachedNeighbors[compacted_data_dst] = wid;
	workgroup_wave_sync_barrier();

	let compacted_data_src = CachedNeighbors[wid];
	let valid_count = WaveActiveCountBits(is_lane_valid);
	data = NodeData.wave_shuffle(data, compacted_data_src);
	cluster_index = WaveShuffle(new_cluster_index, compacted_data_src);
	cluster_index = wid < valid_count ? cluster_index : INVALID_ID;
	return valid_count;
}

groupshared u32 CachedNeighbors[32];
static const u32 SEARCH_RADIUS_BITS = 3;
static const u32 SEARCH_RADIUS = 1 << SEARCH_RADIUS_BITS;
static const u32 PAYLOAD_MASK = (1 << (SEARCH_RADIUS_BITS + 1)) - 1;

u32 find_nearest_neighbor(u32 prim_count, NodeData data) {
	u32 wid = WaveGetLaneIndex();
	CachedNeighbors[wid] = INVALID_ID;
	workgroup_wave_sync_barrier();

	var min_area_index = u32(-1);
	for (u32 r = 1; r <= SEARCH_RADIUS && (wid + r < prim_count); r++) {
		let neighbor = wid + r;
		let neighbor_data = NodeData.wave_shuffle(data, neighbor);
		let new_area = distance_factor(data, neighbor_data);
		// Gives us SEARCH_RADIUS_BITS + 1 bits to store our payload
		let new_area_i = (asuint(new_area) << 1) & ~PAYLOAD_MASK;
		let us_to_neighbor = new_area_i | encode_relative_offset(wid, neighbor);
		let neighbor_to_us = new_area_i | encode_relative_offset(neighbor, wid);
		min_area_index = min(min_area_index, us_to_neighbor);
		atomic_min(CachedNeighbors[neighbor], neighbor_to_us, Scope.Subgroup, Loc.Workgroup);
	}
	atomic_min(CachedNeighbors[wid], min_area_index, Scope.Subgroup, Loc.Workgroup);
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
	let num_valid = WaveActiveCountBits(cluster_index != INVALID_ID);
	return num_valid;
}

void store_cluster_indices(u32 orig_num_prims, u32 cluster_index, u32 l_start) {
	let wid = WaveGetLaneIndex();
	if (wid < orig_num_prims)
		store(Constants.cluster_indices, l_start + wid, cluster_index, Scope.Queue);
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

f32 surface_area(f32x3 aabb_min, f32x3 aabb_max) {
	let d = aabb_max - aabb_min;
	return max(2.f * (d.x * d.y + d.x * d.z + d.y * d.z), 0.f);
}

f32 m_omega(f32x4 cone) {
	let theta_o = cone.w;
	let theta_w = min(theta_o + PI * 0.5f, PI);
	let cos_o = cos(theta_o);
	let sin_o = sin(theta_o);
	return 2.f * PI * (1.f - cos_o) +
		   PI * 0.5f * (2.f * theta_w * sin_o - cos(theta_o - 2.f * theta_w) - 2.f * theta_o * sin_o + cos_o);
}

// SAOH: [Importance Sampling of Many Lights with Adaptive Tree Splitting]
f32 distance_factor(NodeData a, NodeData b) {
	if (any(a.aabb_max < a.aabb_min) || any(b.aabb_max < b.aabb_min))
		return 1e38f;

	let merged = NodeData(a, b);
	let m_a_l = surface_area(a.aabb_min, a.aabb_max);
	let m_a_r = surface_area(b.aabb_min, b.aabb_max);
	let m_a = surface_area(merged.aabb_min, merged.aabb_max);

	let m_omega_l = m_omega(a.cone);
	let m_omega_r = m_omega(b.cone);
	let m_omega = m_omega(merged.cone);

	let e_l = luminance_rec709(a.power);
	let e_r = luminance_rec709(b.power);

	return (e_l * m_a_l * m_omega_l + e_r * m_a_r * m_omega_r) / (m_a * m_omega);
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

NodeData load_node_data(u32 cluster) {
	let is_bvh_node = (cluster & 1) == 1;
	let index = cluster >> 1;
	if (is_bvh_node)
		return load(Constants.bvh_nodes, index, Scope.Queue).data;
	else
		return Constants.build.load_leaf(index);
}
