// https://bruop.github.io/exposure/

import graph;
import graph.util.color;

struct PushConstants {
	u32* histogram;
	f32* exp;
	Tex2D<f32x4> input;
	f32 min_exp;
	f32 inv_exp_range;
	f32 lerp_coeff;
};

[vk::push_constant]
PushConstants Constants;

groupshared u8 bins[256];

f32 lum_to_exp(f32 lum) {
	return log2(lum) + 3.f;	 // log2(lum * 8)
}

u8 bin_exp(f32 exp) {
	if (exp < Constants.min_exp)
		return 0;

	let log = clamp((log2(exp) - Constants.min_exp) * Constants.inv_exp_range, 0.0f, 1.0f);
	return u8(log * 254.f + 1.f);
}

f32 unbin_exp(f32 bin) {
	let log = (bin - 1.f) / 254.f;
	return exp2(log / Constants.inv_exp_range + Constants.min_exp);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(u32x2 id: SV_DispatchThreadID, u32 gid: SV_GroupIndex) {
	bins[gid] = 0;
	workgroup_sync_barrier();

	// TODO: novideo dies if we do atomics on groupshared,
	// so we do terrible, cursed stuff instead.
	let size = Constants.input.size();
	if (id.x < size.x && id.y < size.y) {
		let color = Constants.input.load(id).xyz;
		let exp = lum_to_exp(luminance(color));
		let bin = bin_exp(exp);
		atomic_add(Constants.histogram[bin], 1);
		// bins[gid] = bin;
	}
	return;
	workgroup_sync_barrier();

	u8 bin_counts[256] = {};
	if (gid == 0) {
		for (int i = 0; i < 256; i++) {
			bin_counts[u32(bins[i])]++;
		}
		for (int i = 0; i < 256; i++) {
			bins[i] = bin_counts[i];
		}
	}
	workgroup_sync_barrier();

	atomic_add(Constants.histogram[gid], bins[gid]);
}

groupshared u32 shared_sum[16];

[shader("compute")]
[numthreads(256, 1, 1)]
void average() {
	// TODO: force 32 threads per wave
	let wid = wave_id();
	let wc = WaveGetLaneCount();
	let lid = wid * wc + WaveGetLaneIndex();

	// Each wave sums, then final thread sums the waves
	let count = Constants.histogram[lid];
	let val = count * lid;
	let s = WaveActiveSum(val);
	if (WaveIsFirstLane()) {
		shared_sum[wid] = s;
	}
	workgroup_sync_barrier();

	if (lid == 0) {
		var sum = 0.f;
		for (int i = 0; i < wc; i++) {
			sum += shared_sum[i];
		}

		let size = Constants.input.size();
		let total = f32(size.x * size.y);
		let black_pixels = f32(count);
		let pixels = max(total - black_pixels, 1.f);
		let bin = sum / pixels;
		let exp = unbin_exp(bin);
		*Constants.exp = exp;
		return;

		let last = *Constants.exp;
		let lum = lerp(last, exp, Constants.lerp_coeff);
		*Constants.exp = lum;
	}
}
