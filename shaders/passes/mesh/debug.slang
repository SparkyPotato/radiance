module debug;

import graph;
import passes.visbuffer;
import passes.visbuffer_extra;

public struct DebugImpl : VisbufferDebug {
	STex2D<u32, r32u> overdraw_tex;
	STex2D<u64, r64u> hwsw_tex;

	public void write(u32x2 pos, f32 depth, u32 mode) {
		VisBuffer mask = { depth, mode };
		atomic_max(this.hwsw_tex[pos], mask.encode());
		atomic_add(this.overdraw_tex[pos], 1);
	}

	public u32 overdraw(f32x2 uv) {
		return this.overdraw_tex[this.overdraw_tex.pixel_of_uv(uv)];
	}

	public u32 hwsw(f32x2 uv) {
		return u32(this.hwsw_tex[this.hwsw_tex.pixel_of_uv(uv)]);
	}
}

export struct Debug : VisbufferDebug = DebugImpl;
