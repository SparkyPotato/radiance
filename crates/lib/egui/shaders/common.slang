import graph.types;
import graph.texture;
import graph.sampler;

public struct VertexInput {
	public f32x2 position;
	public f32x2 uv;
	public u32 color;
}

public struct VertexOutput {
	public f32x4 position: SV_Position;
	public f32x2 uv: UV;
	public f32x4 color: COLOR;
}

public struct PushConstants {
	public f32x2 screen_size;
	public VertexInput* vertices;
	public Tex<f32x4, D2> image;
	public Sampler sampler;
}

public [vk::push_constant] PushConstants Constants;
