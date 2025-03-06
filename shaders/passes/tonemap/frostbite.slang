module frostbite;

import graph;
import graph.util;
import graph.util.color;
import passes.tonemap.common;

f32 range_compress(f32 x) {
	return 1.f - exp(-x);
}

f32x3 range_compress(f32x3 x) {
	return f32x3(1.f) - exp(-x);
}

f32x3 range_compress(f32x3 val, f32x3 threshold) {
	let v1 = val;
	let v2 = threshold + (1.f - threshold) * range_compress((val - threshold) / (1.f - threshold));
	return select(val < threshold, v1, v2);
}

struct FrostbiteTonemap : IFunc<f32x3, f32x3> {
	f32x3 operator()(f32x3 color) {
		// TODO: parameterize sdr whitepoint.
		// color *= 200.f;

		let ictcp = rec2020_to_ICtCp(color);
		let sat_amount = pow(smoothstep(1.f, 0.3f, ictcp.x), 1.3f);
		color = ICtCp_to_rec2020(ictcp * f32x3(1.f, sat_amount.xx));

		let linear_segment_end = 0.25f;
		let max_col = max3(color.x, color.y, color.z);
		let mapped_max = range_compress(max_col, linear_segment_end);
		let compressed = color * mapped_max / max_col;
		let per_channel_compressed = range_compress(color, linear_segment_end);
		color = lerp(per_channel_compressed, compressed, 0.6f);

		var ictcp_mapped = rec2020_to_ICtCp(color);
		let post_compress_saturation_boost = 0.3f * smoothstep(1.f, 0.5f, ictcp.x);
		ictcp_mapped.yz =
			lerp(ictcp_mapped.y, ictcp.yz * ictcp_mapped.x / max(1e-3, ictcp.x), post_compress_saturation_boost);
		color = ICtCp_to_rec2020(ictcp_mapped);
		return pq_oetf(color * 200.f);
	}
}

struct PushConstants {
	TonemapInput i;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
f32x4 main(ScreenOutput s) : SV_Target0 {
	return tonemap(s, Constants.i, FrostbiteTonemap());
}
