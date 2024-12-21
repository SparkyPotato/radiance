// https://bruop.github.io/exposure/

import graph;
import graph.util.color;

struct PushConstants {
	u32* histogram;
	f32* avg_lum;
	Tex2D<f32x4> input;
	f32 min_log_lum;
	f32 inv_log_lum_range;
	f32 lerp_coeff;
};

[vk::push_constant]
PushConstants Constants;

groupshared u32 shared_histogram[256];

u32 bin_color(f32x3 color) {
	let lum = luminance(color);
	if (lum < 0.005f)
		return 0;

	let log = clamp((log2(lum) - Constants.min_log_lum) * Constants.inv_log_lum_range, 0.0f, 1.0f);
	return u32(log * 254.f + 1.f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(u32x2 id: SV_DispatchThreadID, u32 gid: SV_GroupIndex) {
	shared_histogram[gid] = 0;
	workgroup_sync_barrier();

	let size = Constants.input.size();
	if (id.x < size.x && id.y < size.y) {
		let color = Constants.input.load(id).xyz;
		let bin = bin_color(color);
		// atomic_add(Constants.histogram[bin], 1);
		shared_histogram[bin] += 1;
	}
	workgroup_sync_barrier();

	atomic_add(Constants.histogram[gid], shared_histogram[gid]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void average() {
	let wid = wave_id();
	let wc = WaveGetLaneCount();
	let lid = wid * wc + WaveGetLaneIndex();

	// Each wave sums, then final thread sums the waves
	let count = Constants.histogram[lid];
	let val = count * lid;
	let val2 = count;
	let s = WaveActiveSum(val);
	let s2 = WaveActiveSum(val2);
	if (WaveIsFirstLane()) {
		shared_histogram[wid] = s;
		shared_histogram[wid + wc] = s2;
	}
	workgroup_sync_barrier();

	if (lid == 0) {
		var sum = 0.f;
		var total = 0.f;
		for (int i = 0; i < wc; i++) {
			sum += shared_histogram[i];
			total += shared_histogram[i + wc];
		}

		let black_pixels = f32(count);
		let pixels = max(total, 1.f);
		let log_avg = sum / pixels - 1.f;
		let scale = 254.f * Constants.inv_log_lum_range;
		let avg = exp2(log_avg / scale + Constants.min_log_lum);

		let last = *Constants.avg_lum;
		let lum = lerp(last, avg, Constants.lerp_coeff);
		*Constants.avg_lum = lum;
	}
}
