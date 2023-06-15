#pragma once

struct Sampler {
    u32 index;

    SamplerState get() {
        return Samplers[this.index];
    }
};
