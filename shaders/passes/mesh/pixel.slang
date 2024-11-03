module pixel;

import graph;
import passes.visbuffer;

struct PushConstants {
	u8 _offset[24];
	VisBufferTex output;
}

[vk::push_constant]
PushConstants Constants;

[shader("pixel")]
void main(u32 data: VisBuffer, f32x4 pos: SV_Position) {
	Constants.output.write(u32x2(pos.xy), pos.z, data, 3);
}
