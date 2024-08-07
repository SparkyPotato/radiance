/// Stolen from https://github.com/GPUOpen-Effects/FidelityFX-SPD/blob/master/ffx-spd/ffx_spd.h

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
};

PUSH PushConstants Constants;

groupshared f32 inter[16][16];
groupshared u32 counter;

f32 load_src(uint2 p) {
	float2 inv = float2(Constants.inv_size);
	float2 coord = p * inv + inv;
	return Constants.depth.sample_mip(Constants.s, coord, 0).x;
}

f32 load(uint2 p) {
	uint w, h, _;
	Coherents[Constants.out5].GetDimensions(w, h);
	uint2 dim = uint2(w, h);
	if (all(p < dim)) {
		return Coherents[NonUniformResourceIndex(Constants.out5)].Load(p);
	} else {
		return 0.f;
	}
}

void store(uint2 p, f32 v, u32 mip) {
	if (mip < 5) {
		uint2 dim = Constants.out1[mip].dimensions();
		if (all(p < dim)) {
			Constants.out1[mip].store(p, v);
		}
	} else if (mip == 5) {
		uint w, h, _;
		Coherents[Constants.out5].GetDimensions(w, h);
		uint2 dim = uint2(w, h);
		if (all(p < dim)) {
			Coherents[NonUniformResourceIndex(Constants.out5)][p] = v;
		}
	} else {
		uint2 dim = Constants.out2[mip - 6].dimensions();
		if (all(p < dim)) {
			Constants.out2[mip - 6].store(p, v);
		}
	}
}

bool should_exit(u32 lid) {
	if (lid == 0) {
		counter = Constants.atomic.atomic_add(0, 1);
	}
	GroupMemoryBarrierWithGroupSync();
	return counter != (Constants.workgroups - 1);
}

f32 reduce4(f32 v0, f32 v1, f32 v2, f32 v3) {
	return min(min(v0, v1), min(v2, v3));
}

f32 reduce_load4(uint2 base) {
	return reduce4(
		load(base + uint2(0, 0)),
		load(base + uint2(0, 1)),
		load(base + uint2(1, 0)),
		load(base + uint2(1, 1))
	);
}

f32 reduce_quad(f32 v) {
	u32 quad = WaveGetLaneIndex() & (~0x3);
	f32 v0 = v;
	f32 v1 = WaveReadLaneAt(v, quad | 1);
	f32 v2 = WaveReadLaneAt(v, quad | 2);
	f32 v3 = WaveReadLaneAt(v, quad | 3);
	return reduce4(v0, v1, v2, v3);
}

void downsample_01(u32 x, u32 y, uint2 gid, u32 lid) {
	f32 v[4];

	uint2 tex = gid * 64 + uint2(x * 2, y * 2);
	uint2 pix = gid * 32 + uint2(x, y);
	v[0] = load_src(tex);
	store(pix, v[0], 0);

	tex = gid * 64 + uint2(x * 2 + 32, y * 2);
	pix = gid * 32 + uint2(x + 16, y);
	v[1] = load_src(tex);
	store(pix, v[1], 0);

	tex = gid * 64 + uint2(x * 2, y * 2 + 32);
	pix = gid * 32 + uint2(x, y + 16);
	v[2] = load_src(tex);
	store(pix, v[2], 0);

	tex = gid * 64 + uint2(x * 2 + 32, y * 2 + 32);
	pix = gid * 32 + uint2(x + 16, y + 16);
	v[3] = load_src(tex);
	store(pix, v[3], 0);

	if (Constants.mips <= 1) return;

	v[0] = reduce_quad(v[0]);
	v[1] = reduce_quad(v[1]);
	v[2] = reduce_quad(v[2]);
	v[3] = reduce_quad(v[3]);

	if ((lid & 3) == 0) {
		store(gid * 16 + uint2(x / 2, y / 2), v[0], 1);
		inter[x / 2][y / 2] = v[0];

		store(gid * 16 + uint2(x / 2 + 8, y / 2), v[1], 1);
		inter[x / 2 + 8][y / 2] = v[1];

		store(gid * 16 + uint2(x / 2, y / 2 + 8), v[2], 1);
		inter[x / 2][y / 2 + 8] = v[2];

		store(gid * 16 + uint2(x / 2 + 8, y / 2 + 8), v[3], 1);
		inter[x / 2 + 8][y / 2 + 8] = v[3];
	}
}

void downsample_2(u32 x, u32 y, uint2 gid, u32 lid, u32 mip) {
	f32 v = inter[x][y];
	v = reduce_quad(v);
	if ((lid & 3) == 0) {
		store(gid * 8 + uint2(x / 2, y / 2), v, mip);
		inter[x + ((y / 2) & 1)][y] = v;
	}
}

void downsample_3(u32 x, u32 y, uint2 gid, u32 lid, u32 mip) {
	if (lid < 64) {
		f32 v = inter[x * 2 + (y & 1)][y * 2];
		v = reduce_quad(v);
		if ((lid & 3) == 0) {
			store(gid * 4 + uint2(x / 2, y / 2), v, mip);
			inter[x * 2 + y / 2][y * 2] = v;
		}
	}
}

void downsample_4(u32 x, u32 y, uint2 gid, u32 lid, u32 mip) {
	if (lid < 16) {
		f32 v = inter[x * 4][y * 4];
		v = reduce_quad(v);
		if ((lid & 3) == 0) {
			store(gid * 2 + uint2(x / 2, y / 2), v, mip);
			inter[x / 2 + y][0] = v;
		}
	}
}

void downsample_5(uint2 gid, u32 lid, u32 mip) {
	if (lid < 4) {
		f32 v = inter[lid][0];
		v = reduce_quad(v);
		if ((lid & 3) == 0) {
			store(gid, v, mip);
		}
	}
}

void downsample_67(u32 x, u32 y) {
	uint2 tex = uint2(x * 4 + 0, y * 4 + 0);
	uint2 pix = uint2(x * 2 + 0, y * 2 + 0);
	f32 v0 = reduce_load4(tex);
	store(pix, v0, 6);

	tex = uint2(x * 4 + 2, y * 4 + 0);
	pix = uint2(x * 2 + 1, y * 2 + 0);
	f32 v1 = reduce_load4(tex);
	store(pix, v1, 6);

	tex = uint2(x * 4 + 0, y * 4 + 2);
	pix = uint2(x * 2 + 0, y * 2 + 1);
	f32 v2 = reduce_load4(tex);
	store(pix, v2, 6);

	tex = uint2(x * 4 + 2, y * 4 + 2);
	pix = uint2(x * 2 + 1, y * 2 + 1);
	f32 v3 = reduce_load4(tex);
	store(pix, v3, 6);

	if (Constants.mips <= 7) return;

	f32 v = reduce4(v0, v1, v2, v3);
	store(uint2(x, y), v, 7);
	inter[x][y] = v;
}

void downsample_next_4(u32 x, u32 y, uint2 gid, u32 lid, u32 base) {
	u32 mips = Constants.mips;

	if (mips <= base + 0) return;
	GroupMemoryBarrierWithGroupSync();
	downsample_2(x, y, gid, lid, base + 0);

	if (mips <= base + 1) return;
	GroupMemoryBarrierWithGroupSync();
	downsample_3(x, y, gid, lid, base + 1);

	if (mips <= base + 2) return;
	GroupMemoryBarrierWithGroupSync();
	downsample_4(x, y, gid, lid, base + 2);

	if (mips <= base + 3) return;
	GroupMemoryBarrierWithGroupSync();
	downsample_5(gid, lid, base + 3);
}

u32 ABfiM(u32 src, u32 ins, u32 bits) {
	u32 mask = (1u << bits) - 1u;
	return (ins & mask) | (src & (~mask));
} 

u32 ABfe(u32 src, u32 off, u32 bits){
	u32 mask = (1u << bits) - 1u;
	return (src >> off) & mask;
}

uint2 remap(u32 a){
	return uint2(
		ABfiM(ABfe(a, 2u, 3u), a, 1u), 
		ABfiM(ABfe(a, 3u, 3u), ABfe(a, 1u, 2u), 2u)
	);
}

[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint lid: SV_GroupIndex) {
	uint2 sub_xy = remap(lid & 63);
	u32 x = sub_xy.x + 8 * ((lid >> 6) & 1);
	u32 y = sub_xy.y + 8 * ((lid >> 7));

	downsample_01(x, y, gid.xy, lid);
	downsample_next_4(x, y, gid.xy, lid, 2);
	if (Constants.mips <= 6) return;
	if (should_exit(lid)) return;
	downsample_67(x, y);
	downsample_next_4(x, y, uint2(0, 0), lid, 8);
}
