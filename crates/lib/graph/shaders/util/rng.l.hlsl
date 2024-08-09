#include "radiance-graph/types.l.hlsl"

static const f32 PI = 3.14159265359f;

// PCG RNG.
struct Rng {
	u32 seed;

	Rng init(u32 id) {
		Rng r;
		r.seed = this.seed + id;
		return r;
	}

	u32 next() {
		u32 state = this.seed;
		this.seed = this.seed * 747796405 + 2891336453;
		u32 word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
		return (word >> 22) ^ word;
	}

	f32 sample() {
		u32 s = this.next();
		return f32(s) / 4294967296.f;
	}

	float2 sample2() {
		return float2(this.sample(), this.sample());
	}

	float2 sample_disk() {
		float2 p = this.sample2() * 2.f - 1.f;
		if (p.x == 0.f && p.y == 0.f) return p;

		f32 theta, r;
		if (abs(p.x) > abs(p.y)) {
			r = p.x;
			theta = PI * (p.y / p.x) / 4.f;
		} else {
			r = p.y;
			theta = PI / 2.f - PI * (p.x / p.y) / 4.f;
		}
		return float2(r * cos(theta), r * sin(theta));
	}

	float3 sample_cos_hemi() {
		float2 p = this.sample_disk();
		f32 y = sqrt(max(0.f, 1.f - p.x * p.x - p.y * p.y));
		return float3(p.x, y, p.y);
	}
};

float3x3 gen_basis(float3 y) {
	float3 other;
	if (y.x < 0.1f && y.y < 0.1f) {
		other = float3(0.f, -y.z, y.y);
	} else {
		other = float3(-y.y, y.x, 0.f);
	}
	other = normalize(other);
	float3 last = cross(other, y);
	float3x3 ret = {
		other.x, y.x, last.x,
		other.y, y.y, last.y,
		other.z, y.z, last.z
	};
	return ret;
}

