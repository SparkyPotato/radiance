module tony_mc_mapface;

import graph;
import graph.util;
import passes.tonemap.common;

static const f32 LUT_SIZE = 48.f;

struct TonyMcMapface : IFunc<f32x3, f32x3> {
	Tex3D<f32x3> lut;
	Sampler sampler;
	f32x3 operator()(f32x3 color) {
		color /= color + 1.f;
		let uv = color * ((LUT_SIZE - 1.f) / LUT_SIZE) + (0.5f / LUT_SIZE);
		return this.lut.sample_mip(this.sampler, uv, 0.f);
	}
}

struct PushConstants {
	TonemapInput i;
	TonyMcMapface map;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
f32x4 main(ScreenOutput input) : SV_Target0 {
	return tonemap(input, Constants.i, Constants.map);
}
