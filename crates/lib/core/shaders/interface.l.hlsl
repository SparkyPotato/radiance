#define PUSH [[vk::push_constant]]

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float16_t f16;
typedef float f32;
typedef double f64;

[[vk::binding(0, 0)]] RWByteAddressBuffer Buffers[];

[[vk::binding(1, 0)]] Texture2D Texture1Ds[];
[[vk::binding(1, 0)]] Texture2D Texture2Ds[];
[[vk::binding(1, 0)]] Texture2D Texture3Ds[];
[[vk::binding(1, 0)]] TextureCube TextureCubes[];

// [[vk::binding(2, 0)]] RWTexture2D RWTexture1Ds[];
// [[vk::binding(2, 0)]] RWTexture2D RWTexture2Ds[];
// [[vk::binding(2, 0)]] RWTexture2D RWTexture3Ds[];

[[vk::binding(3, 0)]] SamplerState Samplers[];

template<typename T>
struct Buf {
    u32 index;

    T load(u32 index) {
        return Buffers[this.index].template Load<T>(sizeof(T) * index);
    }

    void store(u32 index, T value) {
        Buffers[this.index].template Store<T>(sizeof(T) * index, value);
    }
};

template<>
struct Buf<u32> {
    u32 index;

    u32 load(u32 index) {
        return Buffers[this.index].Load<u32>(sizeof(u32) * index);
    }

    void store(u32 index, u32 value) {
        Buffers[this.index].Store<u32>(sizeof(u32) * index, value);
    }

    u32 atomic_add(u32 index, u32 value) {
        u32 ret;
        Buffers[this.index].InterlockedAdd(sizeof(u32) * index, value, ret);
        return ret;
    }
};

struct ByteBuf {
    u32 index;

    RWByteAddressBuffer get() {
        return Buffers[this.index];
    }

    template<typename T>
    T load(u32 offset) {
        return get().template Load<T>(offset);
    }

    u16 load_byte(u32 offset) {
        return get().Load<u16>(offset) & 0xff;
    }
};

struct Sampler {
    u32 index;

    SamplerState get() {
        return Samplers[this.index];
    }
};

struct Tex2D {
    u32 index;

    Texture2D get() {
        return Texture2Ds[this.index];
    }

    float4 sample(Sampler sampler, float2 uv) {
        return get().Sample(sampler.get(), uv);
    }

    float4 sample(Sampler sampler, float2 uv, int2 offset) {
        return get().Sample(sampler.get(), uv, offset);
    }
};
