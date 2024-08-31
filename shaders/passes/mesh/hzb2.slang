module hzb2;

/// Stolen from https://github.com/Themaister/Granite/blob/master/assets/shaders/post/hiz.comp

import graph;
import visbuffer;

struct PushConstants {
	STex<f32, D2, Incoherent, NonUniform> mip5;
	STex<f32, D2, Incoherent, NonUniform> out[6];
	u32 mips;
}

[vk::push_constant]
PushConstants Constants;

groupshared f32 inter[16];
groupshared bool is_last;

f32 reduce(f32 x, f32 y) {
	return min(x, y);
}

f32 reduce(f32x4 v) {
	return reduce(reduce(v.x, v.y), reduce(v.z, v.w));
}

u32x2 mip_size(u32 mip) {
	return Constants.out[mip - 6].size();
}

void store(u32x2 p, u32 mip, f32 v) {
	let dim = mip_size(mip);
	if (any(p >= dim)) {
		return;
	}
	Constants.out[mip - 6].store(p, v);
}

void store4(u32x2 p, u32 mip, f32x4 v) {
	store(p + u32x2(0, 0), mip, v.x);
	store(p + u32x2(1, 0), mip, v.y);
	store(p + u32x2(0, 1), mip, v.z);
	store(p + u32x2(1, 1), mip, v.w);
}

f32x4 fetch2x2(u32x2 p) {
	let minc = Constants.mip5.size() - 1;
	let x = Constants.mip5.load(min(p + u32x2(0, 0), minc));
	let y = Constants.mip5.load(min(p + u32x2(1, 0), minc));
	let z = Constants.mip5.load(min(p + u32x2(0, 1), minc));
	let w = Constants.mip5.load(min(p + u32x2(1, 1), minc));
	return f32x4(x, y, z, w);
}

f32x4x4 fetch4x4(u32x2 p) {
	let x = fetch2x2(p + u32x2(0, 0));
	let y = fetch2x2(p + u32x2(2, 0));
	let z = fetch2x2(p + u32x2(0, 2));
	let w = fetch2x2(p + u32x2(2, 2));
	return f32x4x4(x, y, z, w);
}

f32 reduce_mip_reg(f32x4x4 m, u32x2 p, u32 mip, bool full) {
	let q0 = m[0];
	let q1 = m[1];
	let q2 = m[2];
	let q3 = m[3];
	let res = mip_size(mip);
	var d0 = reduce(q0);
	var d1 = reduce(q1);
	var d2 = reduce(q2);
	var d3 = reduce(q3);

	if (!full) {
		if (p.x + 1 == res.x) {
			d0 = reduce(d0, d1);
			d2 = reduce(d2, d3);
		}
		if (p.y + 1 == res.y) {
			d2 = reduce(d2, d0);
			d3 = reduce(d3, d1);
		}
	}

	let ret = f32x4(d0, d1, d2, d3);
	store4(p, mip, ret);
	return reduce(ret);
}

f32 reduce_mip_simd(u32x2 p, u32 lid, u32 mip, f32 d, bool full) {
	var res = mip_size(mip);
	var horiz = QuadReadAcrossX(d);
	var vert = QuadReadAcrossY(d);
	var diag = QuadReadAcrossDiagonal(d);
	if (!full) {
		bool shoriz = p.x + 1 == res.x;
		bool svert = p.y + 1 == res.y;
		if (shoriz)
			d = reduce(d, horiz);
		if (svert)
			d = reduce(d, vert);
		if (shoriz && svert)
			d = reduce(d, diag);
	}
	store(p, mip, d);

	if (Constants.mips > mip + 1) {
		p >>= 1;
		res = mip_size(mip + 1);
		d = reduce(f32x4(d, horiz, vert, diag));
		horiz = wave_shuffle_xor(d, 0b1000);
		vert = wave_shuffle_xor(d, 0b100);
		diag = wave_shuffle_xor(d, 0b1100);
		if (!full) {
			bool shoriz = p.x + 1 == res.x;
			bool svert = p.y + 1 == res.y;
			if (shoriz)
				d = reduce(d, horiz);
			if (svert)
				d = reduce(d, vert);
			if (shoriz && svert)
				d = reduce(d, diag);
		}
		if ((lid & 3) == 0)
			store(p, mip + 1, d);
	}

	return reduce(f32x4(d, horiz, vert, diag));
}

u32x2 unswizzle(u32 index) {
	let x0 = index & 0b1;
	let y01 = (index >> 1) & 0b11;
	let x12 = (index >> 3) & 0b11;
	let y23 = (index >> 5) & 0b11;
	let x3 = (index >> 7) & 0b1;
	return u32x2(x0 | (x12 << 1) | (x3 << 3), y01 | (y23 << 2));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(u32x2 gid: SV_GroupID) {
	let lid = wave_id() * WaveGetLaneCount() + WaveGetLaneIndex();
	let p = unswizzle(lid);

	let base = p * 4;
	var d = reduce_mip_reg(fetch4x4(base), base >> 1, 6, false);
	if (Constants.mips <= 7)
		return;

	d = reduce_mip_simd(p, lid, 7, d, false);
	if (Constants.mips <= 9)
		return;
	if ((lid & 15) == 0)
		inter[lid >> 4] = d;
	workgroup_sync_barrier();
	if (lid < 16)
		d = reduce_mip_simd(p, lid, 9, inter[lid], false);
	if (Constants.mips <= 11)
		return;

	if (lid == 0)
		store(u32x2(0, 0), 11, d);
}
