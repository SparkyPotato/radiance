module debug;

import graph;
import visbuffer;
import visbuffer_extra;

public struct DebugImpl : VisbufferDebug {
	STex<u32, D2> overdraw_tex;
	STex<u64, D2> hwsw_tex;

	public void write(u32x2 pos, f32 depth, bool is_sw) {
		VisBuffer mask = { depth, (is_sw ? 1 : 2) };
		this.hwsw_tex.atomic_max(pos, mask.encode());
		this.overdraw_tex.atomic_add(pos, 1);
	}
	
	public u32 overdraw(f32x2 uv) {
		return this.overdraw_tex.load(this.overdraw_tex.pixel_of_uv(uv));
	}

	public u32 hwsw(f32x2 uv) {
		return u32(this.hwsw_tex.load(this.hwsw_tex.pixel_of_uv(uv)));
	}
}

export struct Debug : VisbufferDebug = DebugImpl;
