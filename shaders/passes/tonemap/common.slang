module common;

import graph;
import graph.util;

public struct TonemapInput {
	f32* exposure;
	Tex2D<f32x4> src;
	u32 _pad;
}

public f32x4 tonemap(ScreenOutput s, TonemapInput i, IFunc<f32x3, f32x3> f) {
	let pixel = i.src.pixel_of_uv(s.uv);
	let color = i.src.load(pixel).xyz;
	let exposed = color / (exp2(*i.exposure) * 1.2f);
	return f32x4(f(exposed), 1.f);
}
