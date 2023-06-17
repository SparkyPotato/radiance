#pragma once

template<typename T>
struct Buf {
    u32 index;

    T load(u32 i) {
        return Buffers[index].template Load<T>(sizeof(T) * i);
    }

    void store(u32 i, T value) {
        Buffers[index].template Store<T>(sizeof(T) * i, value);
    }
};

template<>
struct Buf<u32> {
    u32 index;

    u32 load(u32 i) {
        return Buffers[index].Load<u32>(sizeof(u32) * i);
    }

    void store(u32 i, u32 value) {
        Buffers[index].Store<u32>(sizeof(u32) * i, value);
    }

    u32 atomic_add(u32 i, u32 value) {
        u32 ret;
        Buffers[index].InterlockedAdd(sizeof(u32) * i, value, ret);
        return ret;
    }
};
