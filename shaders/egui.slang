module egui;

import graph;

struct VertexInput {
	f32x2 position;
	f32x2 uv;
	u32 color;
}

struct VertexOutput {
	f32x4 position : SV_Position;
	f32x2 uv : UV;
	f32x4 color : COLOR;
}

struct PushConstants {
	f32x2 screen_size;
	VertexInput* vertices;
	Tex<f32x4, D2> image;
	Sampler sampler;
}

[vk::push_constant]
PushConstants Constants;

f32x4 unpack_color(u32 color) {
	return f32x4(float(color & 255), float((color >> 8) & 255), float((color >> 16) & 255),
				 float((color >> 24) & 255)) /
		   255.f;
}

f32x4 project_screenspace(f32x2 screenspace) {
	return f32x4(2.f * screenspace / Constants.screen_size - 1.f, 0.f, 1.f);
}

[shader("vertex")]
VertexOutput vertex(u32 vertex: SV_VertexID) {
	let input = Constants.vertices[vertex];

	VertexOutput output;
	output.position = project_screenspace(input.position);
	output.uv = input.uv;
	output.color = unpack_color(input.color);

	return output;
}

f32x3 srgb_gamma_from_linear(f32x3 rgb) {
	let cutoff = rgb < 0.0031308;
	let lower = rgb * 12.92;
	let higher = 1.055 * pow(rgb, 1.0 / 2.4) - 0.055;
	return lerp(higher, lower, f32x3(cutoff));
}

// 0-1 sRGBA gamma  from  0-1 linear
f32x4 srgba_gamma_from_linear(f32x4 rgba) { return f32x4(srgb_gamma_from_linear(rgba.rgb), rgba.a); }

[shader("pixel")]
f32x4 pixel(VertexOutput input) { return Constants.image.sample(Constants.sampler, input.uv) * input.color; }
