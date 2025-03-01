module color;

import graph;

public f32 luminance(f32x3 color) {
	return dot(color, f32x3(0.2125f, 0.7154f, 0.0721f));
}

public f32x3 rec709_to_xyz(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		0.4124564f, 0.3575761f, 0.1804375f,
		0.2126729f, 0.7151522f, 0.0721750f,
		0.0193339f, 0.1191920f, 0.9503041f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 rec2020_to_xyz(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		0.636958f,  0.1446169f, 0.168881f,
		0.2627002f, 0.6779981f, 0.0593017f,
		0.f,        0.0280727f, 1.0609851f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 xyz_to_rec709(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		3.2404542f, -1.5371385f, -0.4985314f,
		-0.9692660f, 1.8760108f, 0.0415560f,
		0.0556434f, -0.2040259f, 1.0572252f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 xyz_to_rec2020(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		1.7166512f, -0.3556708f, -0.2533663f,
		-0.6666844f, 1.6164812f,  0.0157685f,
		0.0176399f, -0.0427706f,  0.9421031
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 xyz_to_Yxy(f32x3 color) {
	let inv = 1.f / dot(color, f32x3(1.f));
	return f32x3(color.y, color.xy * inv);
}

public f32x3 Yxy_to_xyz(f32x3 color) {
	return f32x3(color.x * color.y / color.z, color.x, color.x * (1.f - color.y - color.z) / color.z);
}

public f32x3 rec709_to_Yxy(f32x3 color) {
	return xyz_to_Yxy(rec709_to_xyz(color));
}

public f32x3 Yxy_to_rec709(f32x3 color) {
	return xyz_to_rec709(Yxy_to_xyz(color));
}

public f32x3 rec709_to_rec2020(f32x3 color) {
	return xyz_to_rec2020(rec709_to_xyz(color));
}

public f32x3 rec709_eotf(f32x3 color) {
	let mask = color < 0.081f;
	let a = color / 4.5f;
	let b = pow((color + 0.099f) / 1.099f, 2.22f);
	return select(mask, a, b);
}

public f32x3 pq_oetf(f32x3 color) {
	let m1 = 1305.f / 8192.f;
	let m2 = 2523.f / 32.f;
	let c1 = 107.f / 128.f;
	let c2 = 2413.f / 128.f;
	let c3 = 2392.f / 128.f;

	let y = color / 10000.f;
	let ym1 = pow(y, m1);
	return pow((c1 + c2 * ym1) / (1.f + c3 * ym1), m2);
}
