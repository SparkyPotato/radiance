module sheen;

import graph;
import graph.util.rng;
import passes.bsdf.common;

public f32x3 tint(f32x3 base_color) {
	let lum = dot(f32x3(0.3f, 0.6f, 1.f), base_color);
	return (lum > 0.f) ? base_color / lum : f32x3(1.f);
}

public f32x3 eval_sheen(ShadingParams params, f32x3 wo, f32x3 wm, f32x3 wi) {
	if (params.sheen <= 0.f)
		return f32x3(0.f);

	let h_l = abs(dot(wm, wi));
	let tint = tint(params.base_color);
	return params.sheen * lerp(f32x3(1.f), tint, params.sheen_tint) * shlick_weight(h_l);
}
