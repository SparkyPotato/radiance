#pragma once

#include "types.l.hlsl"
#include "bindings.l.hlsl"

#define UTY Uniform
#define NTY NonUniform
#define GETU() Buffers[this.index]
#define GETN() Buffers[NonUniformResourceIndex(this.index)]

#define TYPED_BUF_DECL(T, U) \
    u32 index; \
    \
    T load(u32 i) {  \
        return GET##U().template Load<T>(sizeof(T) * i); \
    } \
    \
    void store(u32 i, T value) { \
        GET##U().template Store<T>(sizeof(T) * i, value); \
    } \

#define BUF_DECL(U) \
    template<typename T> \
    struct Buf<T, U##TY> { \
        TYPED_BUF_DECL(T, U) \
    };

#define BUF_DECL_U32(U) \
    template<> \
    struct Buf<u32, U##TY> { \
        TYPED_BUF_DECL(u32, U) \
    \
        u32 atomic_add(u32 i, u32 value) { \
            u32 ret; \
            GET##U().InterlockedAdd(sizeof(u32) * i, value, ret); \
            return ret; \
        } \
    };

#define BUF_DECL_BYTES(U) \
    template<> \
    struct Buf<Bytes, U##TY> { \
        u32 index; \
        \
        template<typename T> \
        T load(u32 byte_offset, u32 i) { \
            return GET##U().template Load<T>(byte_offset + sizeof(T) * i); \
        } \
        \
        template<typename T> \
        void store(u32 byte_offset, u32 i, T value) { \
            GET##U().template Store<T>(byte_offset + sizeof(T) * i, value); \
        } \
        \
        u32 atomic_add(u32 byte_offset, u32 i, u32 value) { \
            u32 ret; \
            GET##U().InterlockedAdd(byte_offset + sizeof(u32) * i, value, ret); \
            return ret; \
        } \
    };

template<typename T, typename U = Uniform>
struct Buf {};

BUF_DECL(U)
BUF_DECL(N)

BUF_DECL_U32(U)
BUF_DECL_U32(N)

struct Bytes {};
BUF_DECL_BYTES(U)
BUF_DECL_BYTES(N)

#undef UTY
#undef NTY
#undef GETU
#undef GETN
#undef TYPED_BUF_DECL
#undef BUF_DECL
#undef BUF_DECL_U32
#undef BUF_DECL_BYTES
