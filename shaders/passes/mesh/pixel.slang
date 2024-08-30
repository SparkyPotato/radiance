module pixel;

import graph;
import visbuffer;

struct PushConstants {
	u8 _offset[32];
	VisBufferTex output;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
void main(u32 data: VisBuffer, f32x4 pos: SV_Position) {
	f32 depth = pos.z * pos.w;
	uint2 out_pos = u32x2(pos.xy);
	Constants.output.write(out_pos, depth, data, false);
}
