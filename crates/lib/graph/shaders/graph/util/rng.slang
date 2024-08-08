module rng;

import graph.types;

public static const f32 PI = 3.14159265359f;

// PCG RNG.
public struct Rng {
	u32 seed;

	public func init(id: u32) -> Rng {
		Rng r;
		r.seed = this.seed + id;
		return r;
	}

	[mutating]
	public func next() -> u32 {
		let state = this.seed;
		this.seed = this.seed * 747796405 + 2891336453;
		let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
		return (word >> 22) ^ word;
	}

	[mutating]
	public func sample() -> f32 {
		let s = this.next();
		return f32(s) / 4294967296.f;
	}

	[mutating]
	public func sample2() -> f32x2 {
		return f32x2(this.sample(), this.sample());
	}

	[mutating]
	public func sample_disk() -> f32x2 {
		let p = this.sample2() * 2.f - 1.f;
		if (p.x == 0.f && p.y == 0.f) return p;

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
	public func sample_cos_hemi() -> f32x3 {
		let p = this.sample_disk();
		let y = sqrt(max(0.f, 1.f - p.x * p.x - p.y * p.y));
		return f32x3(p.x, y, p.y);
	}
}

public func gen_basis(y: f32x3) -> float3x3 {
	f32x3 other;
	if (y.x < 0.1f && y.y < 0.1f) {
		other = float3(0.f, -y.z, y.y);
	} else {
		other = float3(-y.y, y.x, 0.f);
	}
	other = normalize(other);
	f32x3 last = cross(other, y);
	float3x3 ret = {
		other.x, y.x, last.x,
		other.y, y.y, last.y,
		other.z, y.z, last.z
	};
	return ret;
}

