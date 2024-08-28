#pragma once

#include "radiance-passes/asset/data.l.hlsl"

struct VisBufferData {
	u32 meshlet_id;
	u32 triangle_id;

	u32 encode() {
		return (meshlet_id << 7) | triangle_id;
	}

	static VisBufferData decode(u32 data) {
		VisBufferData ret;
		ret.meshlet_id = data >> 7;
		ret.triangle_id = data & 0x7f;
		return ret;
	}
};

struct VisBuffer {
	f32 depth;
	u32 data;

	u64 encode() {
		return (u64(asuint(depth)) << 32) | u64(data);
	}

	static VisBuffer decode(u64 data) {
		VisBuffer ret;
		ret.depth = asfloat(u32(data >> 32));
		ret.data = u32(data & 0xffffffff);
		return ret;
	}
};

[[vk::binding(2, 0)]] RWTexture2D<u64> Textures[];
[[vk::binding(2, 0)]] RWTexture2D<u32> OTextures[];

struct VisBufferTex {
	u32 tex;
#ifdef DEBUG
	u32 overdraw;
	u32 hwsw;
#endif

	void write(uint2 pos, f32 depth, u32 data, bool is_sw) {
		VisBuffer d = { depth, data };
		InterlockedMax(Textures[this.tex][pos], d.encode());
#ifdef DEBUG
		VisBuffer mask = { depth, (is_sw ? 1 : 2) };
		InterlockedMax(Textures[this.hwsw][pos], mask.encode());
		InterlockedAdd(OTextures[this.overdraw][pos], 1);
#endif
	}

	uint2 dimensions() {
		u32 w, h;
		Textures[this.tex].GetDimensions(w, h);
		return uint2(w, h);
	}
};

[[vk::binding(1, 0)]] Texture2D<u64> Inputs[];

struct VisBufferRead {
	u32 tex;
#ifdef DEBUG
	Tex2D overdraw_tex;
	u32 hwsw_tex;
#endif

	uint3 uv_to_pos(float2 uv) {
		u32 width, height;
		Inputs[this.tex].GetDimensions(width, height);
		float2 dim = float2(width, height);
		float x = round(uv.x * dim.x - 0.5f);
		float y = round(uv.y * dim.y - 0.5f);
		return uint3(x, y, 0);
	}

	VisBuffer decode(float2 uv) {
		return VisBuffer::decode(Inputs[this.tex].Load(this.uv_to_pos(uv)));
	}

	VisBufferData data(float2 uv) {
		return VisBufferData::decode(this.decode(uv).data);
	}

#ifdef DEBUG
	u32 overdraw(float2 uv) {
		return asuint(this.overdraw_tex.load(this.overdraw_tex.pixel_of_uv(uv)).x);
	}

	u32 hwsw(float2 uv) {
		return u32(Inputs[this.hwsw_tex].Load(this.uv_to_pos(uv)));
	}
#endif
};

struct VertexTransform {
	float4 clip;
	float3 uv;
};

VertexTransform transform_vertex(float4x4 mvp, Vertex vertex) {
	float4 clip = mul(mvp, float4(vertex.position, 1.f));
	float3 ndc = clip.xyz / clip.w;
	float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
	VertexTransform ret = { clip, float3(uv, ndc.z) };
	return ret;
}
