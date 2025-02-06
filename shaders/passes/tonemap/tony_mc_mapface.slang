import graph;
import graph.util;
import passes.tonemap.common;

static const f32 LUT_SIZE = 48.f;

struct TonyMcMapface : Tonemap {
	Tex2D<f32x4> input;
	Tex3D<f32x3> lut;
	Sampler sampler;
	f32 exp;

	Tex2D<f32x4> src() {
		return input;
	}

	f32 exposure() {
		return exp;
	}

	f32x3 map(f32x3 color) {
		color /= color + 1.f;
		let uv = color * ((LUT_SIZE - 1.f) / LUT_SIZE) + (0.5f / LUT_SIZE);
		return this.lut.sample_mip(this.sampler, uv, 0.f);
	}
}

struct PushConstants {
	TonyMcMapface map;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
f32x4 main(ScreenOutput input) : SV_Target0 {
	return tonemap(input, Constants.map);
}
