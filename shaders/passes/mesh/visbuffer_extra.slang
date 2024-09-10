module visbuffer_extra;

import graph;

public interface VisbufferDebug {
	public void write(u32x2 pos, f32 depth, u32 mode);

	public u32 overdraw(f32x2 uv);

	public u32 hwsw(f32x2 uv);
}

public struct Null : VisbufferDebug {
	public void write(u32x2 pos, f32 depth, u32 mode) {}

	public u32 overdraw(f32x2 uv) {
		return 0;
	}

	public u32 hwsw(f32x2 uv) {
		return 0;
	}
}
