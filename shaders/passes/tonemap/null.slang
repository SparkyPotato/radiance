module null;

import graph;
import graph.util;
import graph.util.color;
import passes.tonemap.common;

struct Null : IFunc<f32x3, f32x3> {
	f32x3 operator()(f32x3 color) {
		// TODO: parameterize sdr whitepoint.
		color = pq_oetf(color * 300.f);
		return color;
	}
}

struct PushConstants {
	TonemapInput i;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
f32x4 main(ScreenOutput s) : SV_Target0 {
	return tonemap(s, Constants.i, Null());
}
