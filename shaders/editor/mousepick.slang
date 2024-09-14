import graph;
import passes.visbuffer;

struct PushConstants {
	VisBufferReader read;
	u32x2 pix;
	u32 should_pick;
	u32 _pad;
	u32* ret;
}

[vk::push_constant]
PushConstants Constants;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	let max = 0xffffffff;
	if (Constants.should_pick == 0) {
		*Constants.ret = max;
	} else if (let p = Constants.read.decode(Constants.pix)) {
		*Constants.ret = p.meshlet.instance;
	} else {
		*Constants.ret = max - 1;
	}
}
