import graph.types;
import common;

func unpack_color(color: u32) -> f32x4 {
	return f32x4(
		float(color & 255),
		float((color >> 8) & 255),
		float((color >> 16) & 255),
		float((color >> 24) & 255)
	) / 255.f;
}

func project_screenspace(screenspace: f32x2) -> f32x4 {
	return f32x4(
		2.f * screenspace / Constants.screen_size - 1.f,
		0.f,
		1.f
	);
}

[shader("vertex")]
func main(u32 vertex: SV_VertexID) -> VertexOutput {
	let input = Constants.vertices[vertex];

	VertexOutput output;
	output.position = project_screenspace(input.position);
	output.uv = input.uv;
	output.color = unpack_color(input.color);

	return output;
}
