#pragma once

#include "sampler.l.hlsl"

template<typename T>
struct Tex2D {
    u32 index;

    Texture2D<T> get() {
        return (Texture2D<T>) Texture2Ds[this.index];
    }

    T sample(Sampler sampler, float2 uv) {
        return get().Sample(sampler.get(), uv);
    }

    T load(uint2 pixel, u32 mip = 0) {
        return get().Load(int3(pixel, mip));
    }

    uint2 dimensions(u32 mip = 0) {
        u32 width, height, _;
        get().GetDimensions(mip, width, height, _);
        return uint2(width, height);
    }

    uint2 pixel_of_uv(float2 uv, u32 mip = 0) {
        float2 dim = float2(dimensions(mip));
        float x = round(uv.x * dim.x - 0.5f);
        float y = round(uv.y * dim.y - 0.5f);
        return uint2(x, y);
    }
};
