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

// All inputs are sRGB OETF encoded images with UNORM views (so the shader deals with encoded values, not linear).
// In SDR, the output is also a UNORM view so we can directly write to the swapchain.
// In HDR, convert to linear sRGB, then to linear Rec2020, and finally to PQ Rec2020.

[shader("pixel")]
f32x4 pixel(VertexOutput input) {
	return Constants.image.sample(Constants.sampler, input.uv) * input.color;
}

struct BlitConstants {
	Tex2D<f32x3> image;
}

[vk::push_constant]
BlitConstants BConstants;

[shader("pixel")]
f32x4 blit(ScreenOutput input) {
	var col = BConstants.image.load(BConstants.image.pixel_of_uv(input.uv));
	col = rec709_eotf(col) * 200.f;
	col = rec709_to_rec2020(col);
	col = pq_oetf(col);
	return f32x4(col, 1.f);
}
