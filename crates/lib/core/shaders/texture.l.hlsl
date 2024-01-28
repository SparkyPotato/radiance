#pragma once

#include "types.l.hlsl"
#include "bindings.l.hlsl"
#include "sampler.l.hlsl"

template<typename T, typename U>
struct TextureBase {};
template<typename T>
struct TextureBase<T, Uniform> {
    u32 index;
    Texture2D<T> get() {
        return (Texture2D<T>) Texture2Ds[this.index];
    }
};
template<typename T>
struct TextureBase<T, NonUniform> {
    u32 index;
    Texture2D<T> get() {
        return (Texture2D<T>) Texture2Ds[NonUniformResourceIndex(this.index)];
    }
};

template<typename T, typename U = Uniform>
struct Tex2D: TextureBase<T, U> {
    template<typename S>
    T sample(Sampler<S> sampler, float2 uv) {
        return this.get().Sample(sampler.get(), uv);
    }

    T load(uint2 pixel, u32 mip = 0) {
        return this.get().Load(int3(pixel, mip));
    }

    uint2 dimensions(u32 mip = 0) {
        u32 width, height, _;
        this.get().GetDimensions(mip, width, height, _);
        return uint2(width, height);
    }

    uint2 pixel_of_uv(float2 uv, u32 mip = 0) {
        float2 dim = float2(this.dimensions(mip));
        float x = round(uv.x * dim.x - 0.5f);
        float y = round(uv.y * dim.y - 0.5f);
        return uint2(x, y);
    }
};

template<typename U>
struct StorageTextureBase {};

template<>
struct StorageTextureBase<Uniform> {
    u32 index;
    RWTexture2D<float4> get() {
        return RWTexture2Ds[this.index];
    }
};
template<>
struct StorageTextureBase<NonUniform> {
    u32 index;
    RWTexture2D<float4> get() {
        return RWTexture2Ds[NonUniformResourceIndex(this.index)];
    }
};

template<typename U = Uniform>
struct STex2D: StorageTextureBase<U> {
    float4 load(uint2 pixel, u32 mip = 0) {
        return this.get().Load(int3(pixel, mip));
    }

    void store(uint2 pixel, float4 value) {
        this.get()[pixel] = value;
    }

    uint2 dimensions(u32 mip = 0) {
        u32 width, height, _;
        this.get().GetDimensions(mip, width, height, _);
        return uint2(width, height);
    }

    uint2 pixel_of_uv(float2 uv, u32 mip = 0) {
        float2 dim = float2(this.dimensions(mip));
        float x = round(uv.x * dim.x - 0.5f);
        float y = round(uv.y * dim.y - 0.5f);
        return uint2(x, y);
    }
};
