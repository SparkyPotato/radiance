module egui;

import graph;
import graph.util;
import graph.util.color;

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
	Tex2D<f32x4> image;
	Sampler sampler;
	u32 hdr_image;
}

[vk::push_constant]
PushConstants Constants;

f32x4 unpack_color(u32 color) {
	return f32x4(f32(color & 255), f32((color >> 8) & 255), f32((color >> 16) & 255), f32((color >> 24) & 255)) / 255.f;
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

// Egui requires rendering in Rec709 OETF space, or else blending will be incorrect.
// SDR images are Rec709 OETF encoded with UNORM views so they can be passed through as-is.
// HDR images are PQ Rec2020 encoded, so we need to convert them to Rec709 OETF (out-of-gamut colors are preserved!)

[shader("pixel")]
f32x4 pixel(VertexOutput input) {
	var cola = Constants.image.sample(Constants.sampler, input.uv) * input.color;
	var col = cola.xyz;
	if (Constants.hdr_image != 0) {
		// TODO: parameterize sdr whitepoint.
		col = pq_eotf(col) / 300.f;
		col = rec2020_to_rec709(col);
		col = rec709_oetf(col);
	}
	return f32x4(col, cola.w);
}

struct BlitConstants {
	Tex2D<f32x3> image;
}

[vk::push_constant]
BlitConstants BConstants;

// The blit is only used for HDR output.
// Since the UI is in Rec709 OETF, we must convert to PQ Rec2020 before writing.

[shader("pixel")]
f32x4 blit(ScreenOutput input) {
	var col = BConstants.image.load(BConstants.image.pixel_of_uv(input.uv));
	col = rec709_eotf(col);
	col = rec709_to_rec2020(col) * 300.f;
	col = pq_oetf(col);
	return f32x4(col, 1.f);
}
