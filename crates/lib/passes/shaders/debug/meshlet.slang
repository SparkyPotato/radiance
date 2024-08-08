import graph.types;
import graph.texture;
import graph.util.screen;
import visbuffer;

struct PushConstants {
    Tex<u32, D2> visbuffer;
}

[vk::push_constant] PushConstants Constants;

// Stolen from https://gist.github.com/badboy/6267743 and niagara.
// Thanks copilot.
func hash(a: u32) -> u32 {
	var a = a;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

[shader("pixel")]
public func debug_meshlet(input: ScreenOutput) -> f32x4 {
    let pixel = Constants.visbuffer.pixel_of_uv(input.uv);
    let value = asuint(Constants.visbuffer.load(pixel).x);
    let data = VisBufferData(value);

    if (data.meshlet_pointer_id == 0xffffffff) {
        discard;
    }

    let h = hash(value);
    let color = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255));
    return f32x4(color / 255.0, 1.0);
}
