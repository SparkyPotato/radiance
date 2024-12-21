module color;

import graph;

public f32 luminance(f32x3 color) {
	return dot(color, f32x3(0.2125f, 0.7154f, 0.0721f));
}

public f32x3 rgb_to_xyz(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		0.4124564f, 0.3575761f, 0.1804375f,
		0.2126729f, 0.7151522f, 0.0721750f,
		0.0193339f, 0.1191920f, 0.9503041f
	};
	// clang-format on
	return mul(mat, color);
}

public f32x3 xyz_to_rgb(f32x3 color) {
	// clang-format off
	f32x3x3 mat = {
		3.2404542f, -1.5371385f, -0.4985314f,
		-0.9692660f, 1.8760108f, 0.0415560f,
		0.0556434f, -0.2040259f, 1.0572252f
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

public f32x3 rgb_to_Yxy(f32x3 color) {
	return xyz_to_Yxy(rgb_to_xyz(color));
}

public f32x3 Yxy_to_rgb(f32x3 color) {
	return xyz_to_rgb(Yxy_to_xyz(color));
}
