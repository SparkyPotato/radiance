module sort;

import graph;

// https://github.com/MircoWerner/VkRadixSort
// modified for 64 bit key -> 32 bit value sorting

struct PushConstants {
	u64* in_keys;
	u64* out_keys;
	u32* in_vals;
	u32* out_vals;
	u32* histogram;
	u32 elem_count;
	u32 shift;
	u32 workgroup_count;
	u32 blocks_per_workgroup;
}

[vk::push_constant]
PushConstants Constants;

groupshared u32 SharedHistogram[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void histogram(u32 lid: SV_GroupIndex, u32x3 gid_f: SV_GroupID) {
	let gid = gid_f.x;
	SharedHistogram[lid] = 0;
	workgroup_sync_barrier();

	for (u32 i = 0; i < Constants.blocks_per_workgroup; i++) {
		let elem_id = gid * Constants.blocks_per_workgroup * 256 + i * 256 + lid;
		if (elem_id < Constants.elem_count) {
			let bin = u32(Constants.in_keys[elem_id] >> u64(Constants.shift)) & 255;
			atomic_add(SharedHistogram[bin], 1, Scope.Workgroup);
		}
	}
	workgroup_sync_barrier();

	Constants.histogram[256 * gid + lid] = SharedHistogram[lid];
}

struct BinFlags {
	u32 flags[8];
}

groupshared u32 Sums[8];
groupshared u32 GlobalOffsets[256];
groupshared BinFlags Flags[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void sort(u32x3 tid: SV_DispatchThreadID, u32 lid: SV_GroupIndex, u32x3 gid_f: SV_GroupID) {
	let wid = WaveGetLaneIndex();
	let sid = wave_id();
	let gid = gid_f.x;

	var local_histogram = 0;
	var histogram_count = 0;
	for (u32 j = 0; j < Constants.workgroup_count; j++) {
		let t = Constants.histogram[256 * j + lid];
		local_histogram = select(j == gid, histogram_count, local_histogram);
		histogram_count += t;
	}
	let sum = WaveActiveSum(histogram_count);
	let prefix_sum = WavePrefixSum(histogram_count);
	if (WaveIsFirstLane()) {
		Sums[sid] = sum;
	}
	workgroup_sync_barrier();

	var sums_prefix_sum = 0;
	if (wid < 8) {
		sums_prefix_sum = WavePrefixSum(Sums[wid]);
	}
	sums_prefix_sum = WaveReadLaneAt(sums_prefix_sum, sid);
	let global_histogram = sums_prefix_sum + prefix_sum;
	GlobalOffsets[lid] = global_histogram + local_histogram;

	let flags_bin = lid >> 5;
	let flags_bit = 1 << (lid & 0b11111);
	for (u32 i = 0; i < Constants.blocks_per_workgroup; i++) {
		let elem_id = gid * Constants.blocks_per_workgroup * 256 + i * 256 + lid;

		for (u32 i = 0; i < 8; i++) {
			Flags[lid].flags[i] = 0;
		}
		workgroup_sync_barrier();

		u64 in_key = 0;
		u32 in_val = 0;
		var bin_id = 0;
		var bin_offset = 0;
		if (elem_id < Constants.elem_count) {
			in_key = Constants.in_keys[elem_id];
			in_val = Constants.in_vals[elem_id];
			bin_id = u32(in_key >> u64(Constants.shift)) & 255;
			bin_offset = GlobalOffsets[bin_id];
			atomic_add(Flags[bin_id].flags[flags_bin], flags_bit, Scope.Workgroup);
		}
		workgroup_sync_barrier();

		if (elem_id < Constants.elem_count) {
			var prefix = 0;
			var count = 0;
			for (u32 i = 0; i < 8; i++) {
				let bits = Flags[bin_id].flags[i];
				let full_count = countbits(bits);
				let partial_count = countbits(bits & (flags_bit - 1));
				prefix += select(i < flags_bin, full_count, 0);
				prefix += select(i == flags_bin, partial_count, 0);
				count += full_count;
			}
			Constants.out_keys[bin_offset + prefix] = in_key;
			Constants.out_vals[bin_offset + prefix] = in_val;
			if (prefix == count - 1) {
				atomic_add(GlobalOffsets[bin_id], count, Scope.Workgroup);
			}
		}

		workgroup_sync_barrier();
	}
}
