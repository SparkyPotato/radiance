module common;

import graph;
import graph.util;

public interface Tonemap {
	Tex2D<f32x4> src();

	f32 exposure();

	f32x3 map(f32x3 color);
}

public f32x4 tonemap<T : Tonemap>(ScreenOutput input, T t) {
	let pixel = t.src().pixel_of_uv(input.uv);
	let color = t.src().load(pixel).xyz;
	let exposed = color / (exp2(t.exposure()) * 1.2f);
	return f32x4(t.map(exposed), 1.f);
}
