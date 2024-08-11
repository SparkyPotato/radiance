/// Stolen from https://github.com/Themaister/Granite/blob/master/assets/shaders/post/hiz.comp

#include "radiance-graph/interface.l.hlsl"

[[vk::binding(2, 0)]] globallycoherent RWTexture2D<f32> Coherents[];

struct PushConstants {
	Tex2D depth;
	Sampler s;
	Buf<u32> atomic;
	STex2D<NonUniform> out1[5];
	u32 out5;
	STex2D<NonUniform> out2[6];
	u32 mips;
	u32 workgroups;
	f32 inv_size[2];
	f32 near;
};

PUSH PushConstants Constants;

groupshared f32 inter[16];
groupshared bool is_last;

f32 transform_depth(f32 d) {
	return Constants.near / d;
}

f32 reduce(float4 v) {
	return min(min(v.x, v.y), min(v.z, v.w));
}

uint2 mip_size(u32 mip) {
	if (mip < 5) {
		return Constants.out1[mip].dimensions();
	} else if (mip == 5) {
		uint w, h;
		Coherents[Constants.out5].GetDimensions(w, h);
		return uint2(w, h);
	} else {
		return Constants.out2[mip - 6].dimensions();
	}
}

void store(uint2 p, u32 mip, f32 v) {
	uint2 dim = mip_size(mip);
	if (!all(p < dim)) { return; }
	if (mip < 5) {
		Constants.out1[mip].store(p, v);
	} else if (mip == 5) {
		Coherents[NonUniformResourceIndex(Constants.out5)][p] = v;
	} else {
		Constants.out2[mip - 6].store(p, v);
	}
}

void store4(uint2 p, u32 mip, float4 v) {
	store(p + uint2(0, 0), mip, v.x);
	store(p + uint2(1, 0), mip, v.y);
	store(p + uint2(0, 1), mip, v.z);
	store(p + uint2(1, 1), mip, v.w);
}

f32 fetch(uint2 p) {
	float2 inv = float2(Constants.inv_size);
	float2 coord = p * inv + inv;
	f32 v = Constants.depth.sample_mip(Constants.s, coord, 0).x;
	return transform_depth(v);
}

float4 fetch4x4(uint2 p) {
	f32 x = fetch(p + uint2(0, 0));
	f32 y = fetch(p + uint2(2, 0));
	f32 z = fetch(p + uint2(0, 2));
	f32 w = fetch(p + uint2(2, 2));
	return float4(x, y, z, w);
}

float4 fetch2x2_6(uint2 p) {
	uint2 maxc = mip_size(5) - 1;
	f32 x = Coherents[NonUniformResourceIndex(Constants.out5)].Load(min(p + uint2(0, 0), maxc));
	f32 y = Coherents[NonUniformResourceIndex(Constants.out5)].Load(min(p + uint2(1, 0), maxc));
	f32 z = Coherents[NonUniformResourceIndex(Constants.out5)].Load(min(p + uint2(0, 1), maxc));
	f32 w = Coherents[NonUniformResourceIndex(Constants.out5)].Load(min(p + uint2(1, 1), maxc));
	return float4(x, y, z, w);
}

float4x4 fetch4x4_6(uint2 p) {
	float4 x = fetch2x2_6(p + uint2(0, 0));
	float4 y = fetch2x2_6(p + uint2(2, 0));
	float4 z = fetch2x2_6(p + uint2(0, 2));
	float4 w = fetch2x2_6(p + uint2(2, 2));
	return float4x4(x, y, z, w);
}

f32 reduce_mip_6(float4x4 m, uint2 p, u32 mip) {
	float4 q0 = m[0];
	float4 q1 = m[1];
	float4 q2 = m[2];
	float4 q3 = m[3];
	uint2 res = mip_size(mip);
	f32 d0 = reduce(q0);
	f32 d1 = reduce(q1);
	f32 d2 = reduce(q2);
	f32 d3 = reduce(q3);

	if (p.x + 1 == res.x) {
		d0 = min(d0, d1);
		d2 = min(d2, d3);
	}
	if (p.y + 1 == res.y) {
		d2 = min(d2, d0);
		d3 = min(d3, d1);
	}

	float4 ret = float4(d0, d1, d2, d3);
	store4(p, mip, ret);
	return reduce(ret);
}

f32 reduce_mip_src(float4 m, uint2 p) {
	f32 d0 = m.x;
	f32 d1 = m.y;
	f32 d2 = m.z;
	f32 d3 = m.w;
	uint2 res = mip_size(0);

	if (p.x + 1 == res.x) {
		d0 = min(d0, d1);
		d2 = min(d2, d3);
	}
	if (p.y + 1 == res.y) {
		d2 = min(d2, d0);
		d3 = min(d3, d1);
	}

	float4 ret = float4(d0, d1, d2, d3);
	store4(p, 0, ret);
	return reduce(ret);
}

f32 reduce_mip_simd(uint2 p, u32 lid, u32 mip, f32 d, bool full) {
	uint2 res = mip_size(mip);
	
	f32 horiz = QuadReadAcrossX(d);
	f32 vert = QuadReadAcrossY(d);
	f32 diag = QuadReadAcrossDiagonal(d);
	if (!full) {
		bool shoriz = p.x + 1 == res.x;
		bool svert = p.y + 1 == res.y;
		if (shoriz) d = min(d, horiz);
		if (svert) d = min(d, vert);
		if (shoriz && svert) d = min(d, diag);
	}
	store(p, mip, d);

	if (Constants.mips > mip + 1) {
		p >>= 1;
		res = mip_size(mip + 1);
		d = reduce(float4(d, horiz, vert, diag));
		horiz = WaveReadLaneAt(d, lid ^ 0b1000);
		vert = WaveReadLaneAt(d, lid ^ 0b100);
		diag = WaveReadLaneAt(d, lid ^ 0b1100);
		if (!full) {
			bool shoriz = p.x + 1 == res.x;
			bool svert = p.y + 1 == res.y;
			if (shoriz) d = min(d, horiz);
			if (svert) d = min(d, vert);
			if (shoriz && svert) d = min(d, diag);
		}
		if ((lid & 3) == 0) store(p, mip + 1, d);
	}

	return reduce(float4(d, horiz, vert, diag));
}

u32 bitfield_insert(u32 src, u32 ins, u32 off, u32 bits) {
	u32 mask = ((1u << bits) - 1u) << off;
	return (ins & mask) | (src & (~mask));
} 

u32 bitfield_extract(u32 src, u32 off, u32 bits){
	u32 mask = (1u << bits) - 1u;
	return (src >> off) & mask;
}

uint2 unswizzle(u32 index) {
	u32 x0 = bitfield_extract(index, 0, 1);
	u32 y01 = bitfield_extract(index, 1, 2);
	u32 x12 = bitfield_extract(index, 3, 2);
	u32 y23 = bitfield_extract(index, 5, 2);
	u32 x3 = bitfield_extract(index, 7, 1);
	return uint2(
		bitfield_insert(bitfield_insert(x0, x12, 1, 2), x3, 3, 1), 
		bitfield_insert(y01, y23, 2, 2)
	);
}

[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint lid: SV_GroupIndex) {
	uint2 p = unswizzle(lid);
	
	uint2 base = gid.xy * 64 + p * 4;
	float4 m = fetch4x4(base);
	f32 d = reduce_mip_src(m, base >> 1);
	if (Constants.mips <= 1) return;
	return;

	d = reduce_mip_simd(base >> 1, lid, 1, d, true);
	if (Constants.mips <= 3) return;

	if ((lid & 15) == 0) inter[lid >> 4] = d;
	AllMemoryBarrierWithGroupSync();

	if (lid < 16) d = reduce_mip_simd(gid.xy * 4 + p, lid, 3, inter[lid], true);

	if (Constants.mips <= 5) return;
	if (lid == 0) store(gid.xy, 5, d);
	if (Constants.mips <= 6) return;

	AllMemoryBarrierWithGroupSync();
	if (lid == 0) is_last = Constants.atomic.atomic_add(0, 1) + 1 == Constants.workgroups;
	AllMemoryBarrierWithGroupSync();
	if (!is_last) return;

	base = p * 4;
	d = reduce_mip_6(fetch4x4_6(base), base >> 1, 6);
	if (Constants.mips <= 7) return;

	d = reduce_mip_simd(p, lid, 7, d, false);
	if (Constants.mips <= 9) return;
	if ((lid & 15) == 0) inter[lid >> 4] = d;
	AllMemoryBarrierWithGroupSync();

	if (lid < 16) d = reduce_mip_simd(p, lid, 9, inter[lid], false);
	if (Constants.mips <= 11) return;
	if (lid == 0) store(uint2(0, 0), 11, d);
}
