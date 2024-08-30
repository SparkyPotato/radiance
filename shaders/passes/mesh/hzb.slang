module hzb;

/// Stolen from https://github.com/Themaister/Granite/blob/master/assets/shaders/post/hiz.comp

import graph;
import visbuffer;

struct PushConstants {
	u32* atomic;
	VisBufferTex visbuffer;
	STex<f32, D2, Incoherent, NonUniform> out1[5];
	STex<f32, D2, Incoherent, NonUniform> out5;
	STex<f32, D2, Incoherent, NonUniform> out2[6];
	u32 mips;
	u32 workgroups;
}

[vk::push_constant]
PushConstants Constants;

groupshared f32 inter[16];
groupshared bool is_last;

f32 reduce(f32x4 v) {
	return min(min(v.x, v.y), min(v.z, v.w));
}

u32x2 mip_size(u32 mip) {
	if (mip < 5) {
		return Constants.out1[mip].size();
	} else if (mip == 5) {
		return Constants.out5.size();
	} else {
		return Constants.out2[mip - 6].size();
	}
}

void store(u32x2 p, u32 mip, f32 v) {
	let dim = mip_size(mip);
	if (!all(p < dim)) {
		return;
	}
	if (mip < 5) {
		Constants.out1[mip].store(p, v);
	} else if (mip == 5) {
		Constants.out5.store(p, v);
	} else {
		Constants.out2[mip - 6].store(p, v);
	}
}

void store4(u32x2 p, u32 mip, f32x4 v) {
	store(p + u32x2(0, 0), mip, v.x);
	store(p + u32x2(1, 0), mip, v.y);
	store(p + u32x2(0, 1), mip, v.z);
	store(p + u32x2(1, 1), mip, v.w);
}

f32 fetch(u32x2 p) {
	return Constants.visbuffer.decode(p).depth;
}

f32 fetch2x2(u32x2 p) {
	return reduce(
		f32x4(fetch(p + u32x2(0, 0)), fetch(p + u32x2(1, 0)), fetch(p + u32x2(0, 1)), fetch(p + u32x2(1, 1))));
}

f32x4 fetch4x4(u32x2 p) {
	let x = fetch(p + u32x2(0, 0));
	let y = fetch(p + u32x2(2, 0));
	let z = fetch(p + u32x2(0, 2));
	let w = fetch(p + u32x2(2, 2));
	return f32x4(x, y, z, w);
}

f32x4 fetch2x2_6(u32x2 p) {
	let minc = mip_size(5) - 1;
	let x = Constants.out5.load(min(p + u32x2(0, 0), minc));
	let y = Constants.out5.load(min(p + u32x2(1, 0), minc));
	let z = Constants.out5.load(min(p + u32x2(0, 1), minc));
	let w = Constants.out5.load(min(p + u32x2(1, 1), minc));
	return f32x4(x, y, z, w);
}

f32x4x4 fetch4x4_6(u32x2 p) {
	let x = fetch2x2_6(p + u32x2(0, 0));
	let y = fetch2x2_6(p + u32x2(2, 0));
	let z = fetch2x2_6(p + u32x2(0, 2));
	let w = fetch2x2_6(p + u32x2(2, 2));
	return f32x4x4(x, y, z, w);
}

f32 reduce_mip_6(f32x4x4 m, u32x2 p, u32 mip) {
	let q0 = m[0];
	let q1 = m[1];
	let q2 = m[2];
	let q3 = m[3];
	let res = mip_size(mip);
	var d0 = reduce(q0);
	var d1 = reduce(q1);
	var d2 = reduce(q2);
	var d3 = reduce(q3);

	if (p.x + 1 == res.x) {
		d0 = min(d0, d1);
		d2 = min(d2, d3);
	}
	if (p.y + 1 == res.y) {
		d2 = min(d2, d0);
		d3 = min(d3, d1);
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
			d = min(d, horiz);
		if (svert)
			d = min(d, vert);
		if (shoriz && svert)
			d = min(d, diag);
	}
	store(p, mip, d);

	if (Constants.mips > mip + 1) {
		p >>= 1;
		res = mip_size(mip + 1);
		d = reduce(float4(d, horiz, vert, diag));
		horiz = wave_shuffle_xor(d, 0b1000);
		vert = wave_shuffle_xor(d, 0b100);
		diag = wave_shuffle_xor(d, 0b1100);
		if (!full) {
			bool shoriz = p.x + 1 == res.x;
			bool svert = p.y + 1 == res.y;
			if (shoriz)
				d = min(d, horiz);
			if (svert)
				d = min(d, vert);
			if (shoriz && svert)
				d = min(d, diag);
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
	return uint2(x0 | (x12 << 1) | (x3 << 3), y01 | (y23 << 2));
}

f32 reduce_mip_src(u32x2 base) {
	var m = fetch4x4(base);
	let p = base >> 1;

	let res = mip_size(0);
	if (p.x + 1 == res.x) {
		m.x = min(m.x, m.y);
		m.z = min(m.z, m.w);
	}
	if (p.y + 1 == res.y) {
		m.z = min(m.z, m.z);
		m.w = min(m.w, m.y);
	}

	store4(p, 0, m);
	return reduce(m);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(u32x2 gid: SV_GroupID) {
	let lid = wave_id() * WaveGetLaneCount() + WaveGetLaneIndex();
	let p = unswizzle(lid);

	var base = gid * 64 + p * 4;
	var d = reduce_mip_src(base);
	if (Constants.mips <= 1)
		return;

	d = reduce_mip_simd(base >> 2, lid, 1, d, true);
	if (Constants.mips <= 3)
		return;

	if ((lid & 15) == 0)
		inter[lid >> 4] = d;
	all_sync_barrier();
	if (lid < 16)
		d = reduce_mip_simd(gid * 4 + p, lid, 3, inter[lid], true);
	if (Constants.mips <= 5)
		return;

	if (lid == 0)
		store(gid.xy, 5, d);
	if (Constants.mips <= 6)
		return;

	all_sync_barrier();
	if (lid == 0)
		is_last = atomic_add(*Constants.atomic, 1) + 1 == Constants.workgroups;
	all_sync_barrier();
	if (!is_last)
		return;

	base = p * 4;
	d = reduce_mip_6(fetch4x4_6(base), base >> 1, 6);
	if (Constants.mips <= 7)
		return;

	d = reduce_mip_simd(p, lid, 7, d, false);
	if (Constants.mips <= 9)
		return;
	if ((lid & 15) == 0)
		inter[lid >> 4] = d;
	all_sync_barrier();
	if (lid < 16)
		d = reduce_mip_simd(p, lid, 9, inter[lid], false);
	if (Constants.mips <= 11)
		return;

	if (lid == 0)
		store(uint2(0, 0), 11, d);
}
