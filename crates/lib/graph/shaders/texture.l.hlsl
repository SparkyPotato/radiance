#pragma once

#include "types.l.hlsl"
#include "bindings.l.hlsl"
#include "sampler.l.hlsl"

#define UTY Uniform
#define NTY NonUniform
#define GETU() Texture2Ds[this.index]
#define GETN() Texture2Ds[NonUniformResourceIndex(this.index)]

#define TEX_DECL(U) \
template<> \
struct Tex2D<U##TY> { \
    u32 index; \
    \
    template<typename S> \
    float4 sample(Sampler<S> sampler, float2 uv) { \
        return GET##U().Sample(sampler.get(), uv); \
    } \
    \
    template<typename S> \
    float4 sample_mip(Sampler<S> sampler, float2 uv, f32 mip) { \
        return GET##U().SampleLevel(sampler.get(), uv, mip); \
    } \
    \
    float4 load(uint2 pixel, u32 mip = 0) { \
        return GET##U().Load(int3(pixel, mip)); \
    } \
    \
    uint2 dimensions(u32 mip = 0) { \
        u32 width, height, _; \
        GET##U().GetDimensions(mip, width, height, _); \
        return uint2(width, height); \
    } \
    \
    uint2 pixel_of_uv(float2 uv, u32 mip = 0) { \
        float2 dim = float2(this.dimensions(mip)); \
        float x = round(uv.x * dim.x - 0.5f); \
        float y = round(uv.y * dim.y - 0.5f); \
        return uint2(x, y); \
    } \
};

template<typename U = Uniform>
struct Tex2D {};

TEX_DECL(U)
TEX_DECL(N)

template<typename U = Uniform> 
struct OTex2D {
    Tex2D<U> inner;

    bool valid() { return this.inner.index != 0; }

    template<typename S>
    float4 sample(Sampler<S> sampler, float2 uv, float4 or_else = 0.f) {
        if (this.valid()) return this.inner.sample(sampler, uv);
        else return or_else;
    }

    template<typename S>
    float4 sample_mip(Sampler<S> sampler, float2 uv, f32 mip, float4 or_else = 0.f) {
        if (this.valid()) return this.inner.sample_mip(sampler, uv, mip);
        else return or_else;
    }

    float4 load(uint2 pixel, u32 mip = 0, float4 or_else = 0.f) {
        if (this.valid()) return this.inner.load(pixel, mip);
        else return or_else;
    }

    uint2 dimensions(u32 mip = 0) {
        if (this.valid()) return this.inner.dimensions();
        else return uint2(0, 0);
    }
};

#define UTY Uniform
#define NTY NonUniform
#define GETU() RWTexture2Ds[this.index]
#define GETN() RWTexture2Ds[NonUniformResourceIndex(this.index)]

template<typename U = Uniform>
struct STex2D {};

#define STEX_DECL(U) \
    template<> \
    struct STex2D<U##TY> { \
        u32 index; \
        \
        float4 load(uint2 pixel) { \
            return GET##U().Load(int2(pixel)); \
        } \
        \
        void store(uint2 pixel, float4 value) { \
            GET##U()[pixel] = value; \
        } \
        \
        uint2 dimensions() { \
            u32 width, height; \
            GET##U().GetDimensions(width, height); \
            return uint2(width, height); \
        } \
        \
        uint2 pixel_of_uv(float2 uv) { \
            float2 dim = float2(this.dimensions()); \
            float x = round(uv.x * dim.x - 0.5f); \
            float y = round(uv.y * dim.y - 0.5f); \
            return uint2(x, y); \
        } \
    };
STEX_DECL(U)
STEX_DECL(N)

#undef STEX_DECL
#undef UTY
#undef NTY
#undef GETU
#undef GEN
