module screen;

import graph.types;

public struct ScreenOutput {
	public f32x4 position: SV_Position;
	public f32x2 uv: UV;
}

[shader("vertex")]
public func screen(u32 vertex: SV_VertexID) -> ScreenOutput {
	let uv = f32x2((vertex << 1) & 2, vertex & 2);
	let position = f32x4(uv * 2.0f - 1.0f, 0.0f, 1.0f);

	ScreenOutput ret = { position, uv };
	return ret;
}
