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

u8 bin_lum(f32 lum) {
	if (lum < 0.001f)
		return 0;

	let exp = log2(lum) + 3.f;	// log2(lum * 8)
	let log = clamp((exp - Constants.min_exp) * Constants.inv_exp_range, 0.0f, 1.0f);
	return u8(log * 254.f + 1.f);
}

f32 unbin_exp(f32 bin) {
	let log = (bin - 1.f) / 254.f;
	return log / Constants.inv_exp_range + Constants.min_exp;
}

groupshared u8 bins[256];
groupshared u8 bin_counts[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(u32x2 id: SV_DispatchThreadID, u32 gid: SV_GroupIndex) {
	// TODO: novideo dies if we do atomics on groupshared,
	// so we do terrible, cursed stuff instead.

	let size = Constants.input.size();
	if (id.x < size.x && id.y < size.y) {
		let color = Constants.input.load(id).xyz;
		let bin = bin_lum(luminance(color));
		atomic_add(Constants.histogram[bin], 1);
	}
}
