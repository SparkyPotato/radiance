module color;

import graph;

// TODO: Make a color type generic over a colorspace.

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

public f32x3 lms_to_xyz(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		2.07018f, -1.326457f, 0.206616f,
		0.364988f, 0.680467f, -0.045422f,
		-0.0495955f, -0.0494212f, 1.187996f
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

public f32x3 xyz_to_lms(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		0.3592f, 0.6976f, -0.0358f,
		-0.1922f, 1.1004f, 0.0755f,
		0.007f, 0.0749f, 0.8434f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 rec2020_to_ICtCp(f32x3 color) {
	color = xyz_to_lms(rec2020_to_xyz(color));
	color = pq_oetf(color * 200.f);
	// clang-format off
	f32x3x3 mat = {
		0.5f, 0.5f, 0.f,
		1.6137f, -3.3234f, 1.7097f,
		4.378f, -4.2455f, -0.1325f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 ICtCp_to_rec2020(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		1.f, 0.008605f, 0.1110356f,
		1.f, -0.008605f, -0.1110356f,
		1.f, 0.56004886f, -0.320637f
	};
	// clang-format on
	color = mul(mat, color);
	color = pq_eotf(color) / 200.f;
	return xyz_to_rec2020(lms_to_xyz(color));
}

public f32x3 xyz_to_Yxy(f32x3 color) {
	let inv = 1.f / dot(color, f32x3(1.f));
	return f32x3(color.y, color.xy * inv);
}

public f32x3 Yxy_to_xyz(f32x3 color) {
	return f32x3(color.x * color.y / color.z, color.x, color.x * (1.f - color.y - color.z) / color.z);
}

public f32x3 rec709_to_rec2020(f32x3 color) {
	return xyz_to_rec2020(rec709_to_xyz(color));
}

public f32x3 rec2020_to_rec709(f32x3 color) {
	return xyz_to_rec709(rec2020_to_xyz(color));
}

public f32x3 rec709_oetf(f32x3 color) {
	let mask = color < 0.018f;
	let a = color * 4.5f;
	let b = 1.099f * pow(color, 0.45f) - 0.099f;
	return select(mask, a, b);
}

public f32x3 rec709_eotf(f32x3 color) {
	let mask = color < 0.081f;
	let a = color / 4.5f;
	let b = pow((color + 0.099f) / 1.099f, 2.22f);
	return select(mask, a, b);
}

static const f32 M1 = 1305.f / 8192.f;
static const f32 M2 = 2523.f / 32.f;
static const f32 C1 = 107.f / 128.f;
static const f32 C2 = 2413.f / 128.f;
static const f32 C3 = 2392.f / 128.f;

public f32x3 pq_oetf(f32x3 color) {
	let y = color / 10000.f;
	let ym1 = pow(y, M1);
	return pow((C1 + C2 * ym1) / (1.f + C3 * ym1), M2);
}

public f32x3 pq_eotf(f32x3 color) {
	let ym2 = pow(color, 1.f / M2);
	return 10000.f * pow((max(ym2 - C1, 0.f)) / (C2 - C3 * ym2), 1.f / M1);
}

public f32 luminance_rec709(f32x3 color) {
	return rec709_to_xyz(color).y;
}

public f32 luminance_rec2020(f32x3 color) {
	return rec2020_to_xyz(color).y;
}
