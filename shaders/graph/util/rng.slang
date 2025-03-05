module rng;

import graph;

public struct Sample<T> {
	public T sample;
	public f32 pdf;
}

// PCG
public struct Rng {
	private u32 seed;

	public Rng init_at(u32x2 pix) {
		Rng r;
		r.seed = this.seed + pix.x * 10000 + pix.y;
		return r;
	}

	[mutating]
	u32 next() {
		let state = this.seed;
		this.seed = this.seed * 747796405 + 2891336453;
		let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
		return (word >> 22) ^ word;
	}

	[mutating]
	public f32 sample() {
		let s = this.next();
		return f32(s) / 4294967296.f;
	}

	[mutating]
	public f32x2 sample2() {
		return f32x2(this.sample(), this.sample());
	}

	[mutating]
	public f32x2 sample_disk() {
		let p = this.sample2() * 2.f - 1.f;
		if (p.x == 0.f && p.y == 0.f)
			return p;

		f32 theta, r;
		if (abs(p.x) > abs(p.y)) {
			r = p.x;
			theta = PI * (p.y / p.x) / 4.f;
		} else {
			r = p.y;
			theta = PI / 2.f - PI * (p.x / p.y) / 4.f;
		}
		return f32x2(r * cos(theta), r * sin(theta));
	}

	[mutating]
	public f32x3 sample_cos_hemi() {
		let p = this.sample_disk();
		let z = sqrt(max(0.f, 1.f - p.x * p.x - p.y * p.y));
		return float3(p.x, p.y, z);
	}
}

public f32 pow_heuristic_1(f32 f, f32 g) {
	return f * f / (f * f + g * g);
}

public f32 bal_heuristic_1(f32 f, f32 g) {
	return f / (f + g);
}
