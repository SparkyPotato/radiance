module visbuffer;

import graph;
import passes.asset;
import visbuffer_extra;

extern struct Debug : VisbufferDebug = Null;

public struct VisBufferData {
	public u32 meshlet_id;
	public u32 triangle_id;

	public u32 encode() {
		return (this.meshlet_id << 7) | this.triangle_id;
	}

	public __init(u32 data) {
		VisBufferData ret;
		ret.meshlet_id = data >> 7;
		ret.triangle_id = data & 0x7f;
		return ret;
	}
}

public struct VisBuffer {
	public f32 depth;
	public u32 data;

	public u64 encode() {
		return (u64(asuint(this.depth)) << 32) | u64(this.data);
	}

	public __init(u64 data) {
		VisBuffer ret;
		ret.depth = asfloat(u32(data >> 32));
		ret.data = u32(data & 0xffffffff);
		return ret;
	}
}

public struct VisBufferTex {
	STex<u64, D2> tex;
	Debug dbg;

	public void write(u32x2 pos, f32 depth, u32 data, bool is_sw) {
		VisBuffer d = { depth, data };
		this.tex.atomic_max(pos, d.encode());
		this.dbg.write(pos, depth, is_sw);
	}
	
	public VisBuffer decode(f32x2 uv) {
		return this.decode(this.tex.pixel_of_uv(uv));
	}

	public VisBuffer decode(u32x2 p) {
		return VisBuffer(this.tex.load(p));
	}

	public VisBufferData data(f32x2 uv) {
		return VisBufferData(this.decode(uv).data);
	}

	public u32 overdraw(f32x2 uv) {
		return this.dbg.overdraw(uv);
	}

	public u32 hwsw(f32x2 uv) {
		return this.dbg.hwsw(uv);
	}

	public u32x2 size() {
		return this.tex.size();
	}
};

public struct VertexTransform {
	public f32x4 clip;
	public f32x3 uv;
};

public VertexTransform transform_vertex(f32x4x4 mvp, Vertex vertex) {
	let clip = mul(mvp, f32x4(vertex.position, 1.f));
	let ndc = clip.xyz / clip.w;
	let uv = ndc.xy * f32x2(0.5f, -0.5f) + 0.5f;
	return { clip, f32x3(uv, ndc.z) };
}
