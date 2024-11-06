module debug;

import graph;
import passes.visbuffer;
import passes.visbuffer_extra;

public struct DebugImpl : VisbufferDebug {
	STex2D<u32> overdraw_tex;
	STex2D<u64> hwsw_tex;

	public void write(u32x2 pos, f32 depth, u32 mode) {
		VisBuffer mask = { depth, mode };
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
