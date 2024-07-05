#pragma once

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

struct Uniform {};
struct NonUniform {};
