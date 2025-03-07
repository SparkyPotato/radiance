module exposure;

// https://bruop.github.io/exposure/

import graph;
import graph.util.color;

struct HPushConstants {
	u32* histogram;
	Tex2D<f32x4> input;
	f32 min_exp;
	f32 inv_exp_range;
};

[vk::push_constant]
HPushConstants HConstants;

u8 bin_lum(f32 lum) {
	if (lum < 0.001f)
		return 0;

	let exp = log2(lum) + 3.f;	// log2(lum * 8)
	let log = clamp((exp - HConstants.min_exp) * HConstants.inv_exp_range, 0.0f, 1.0f);
	return u8(log * 254.f + 1.f);
}

groupshared u32 shared_histogram[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(u32x2 id: SV_DispatchThreadID, u32 gid: SV_GroupIndex, u32x2 group: SV_GroupID) {
	shared_histogram[gid] = 0;
	workgroup_sync_barrier();

	let size = HConstants.input.size();
	if (id.x < size.x && id.y < size.y) {
		let color = HConstants.input.load(id).xyz;
		let bin = u32(bin_lum(luminance_rec2020(color)));
		atomic_add(shared_histogram[bin], 1, Scope.Workgroup, Loc.Workgroup);
	}
	workgroup_sync_barrier();

	atomic_add(HConstants.histogram[gid], shared_histogram[gid]);
}

struct EPushConstants {
	u32* histogram;
	f32* exposure;
	f32 histogram_min;
	f32 histogram_max;
	f32 compensation;
	f32 min_exp;
	f32 exp_range;
	f32 lerp_coeff;
};

[vk::push_constant]
EPushConstants EConstants;

groupshared f32 sum_histogram[256];
groupshared f32 count_histogram[256];

f32 brightness_compensation(f32 exp) {
	let lum = exp2(2.0 * exp + 7.0);
	let key = 1.03 - 2.0 / (2.0 + log10(lum + 1.0));
	return 6.0 * key - 2.5;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void exposure(u32 gid: SV_GroupIndex) {
	sum_histogram[gid] = gid == 0 ? 0.f : EConstants.histogram[gid];
	workgroup_sync_barrier();

	if (gid == 0) {
		var prev = 0.f;
		for (var i = 1; i < 256; i++) {
			let curr = sum_histogram[i] + prev;
			sum_histogram[i] = curr;
			prev = curr;
		}
	}
	workgroup_sync_barrier();

	let sum = sum_histogram[255];
	let min_sum = sum * EConstants.histogram_min;
	let max_sum = sum * EConstants.histogram_max;

	let prev_val = gid == 0 ? 0 : sum_histogram[gid - 1];
	let us_val = sum_histogram[gid];
	let count = clamp(us_val, min_sum, max_sum) - clamp(prev_val, min_sum, max_sum);

	workgroup_sync_barrier();
	sum_histogram[gid] = count * f32(gid);
	count_histogram[gid] = count;
	workgroup_sync_barrier();

	for (var cutoff = 128; cutoff > 0; cutoff >>= 1) {
		// gid 0 is not alive.
		if (gid < cutoff) {
			sum_histogram[gid] += sum_histogram[gid + cutoff];
			count_histogram[gid] += count_histogram[gid + cutoff];
		}
		workgroup_sync_barrier();
	}

	if (gid == 0) {
		let hist_sum = sum_histogram[0];
		let hist_count = count_histogram[0];
		f32 exposure;
		if (hist_count == 0.f) {
			exposure = 0.f;
		} else {
			let exp_bin = hist_sum / hist_count;
			let log = (exp_bin - 1.0) / 254.0;
			exposure = log * EConstants.exp_range + EConstants.min_exp;
		}
		let target = exposure - (brightness_compensation(exposure) + EConstants.compensation);
		EConstants.exposure[0] = lerp(EConstants.exposure[0], target, EConstants.lerp_coeff);
		EConstants.exposure[1] = target;
		EConstants.exposure[2] = exposure;
	}
}
