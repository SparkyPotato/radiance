#pragma once

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
