#include "radiance-core/types.l.hlsl"

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
		return f32(s) / 4294967295.f;
	}
};

