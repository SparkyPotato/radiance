implementing graph;

public typedef uint8_t u8;
public typedef uint8_t2 u8x2;
public typedef uint8_t3 u8x3;
public typedef uint8_t4 u8x4;

public typedef uint16_t u16;
public typedef uint16_t2 u16x2;
public typedef uint16_t3 u16x3;
public typedef uint16_t4 u16x4;

public typedef uint32_t u32;
public typedef uint32_t2 u32x2;
public typedef uint32_t3 u32x3;
public typedef uint32_t4 u32x4;

public typedef uint64_t u64;
public typedef uint64_t2 u64x2;
public typedef uint64_t3 u64x3;
public typedef uint64_t4 u64x4;

public typedef int8_t i8;
public typedef int8_t2 i8x2;
public typedef int8_t3 i8x3;
public typedef int8_t4 i8x4;

public typedef int16_t i16;
public typedef int16_t2 i16x2;
public typedef int16_t3 i16x3;
public typedef int16_t4 i16x4;

public typedef int32_t i32;
public typedef int32_t2 i32x2;
public typedef int32_t3 i32x3;
public typedef int32_t4 i32x4;

public typedef int64_t i64;
public typedef int64_t2 i64x2;
public typedef int64_t3 i64x3;
public typedef int64_t4 i64x4;

public typedef float16_t f16;
public typedef float16_t2 f16x2;
public typedef float16_t3 f16x3;
public typedef float16_t4 f16x4;

public typedef float32_t f32;
public typedef float32_t2 f32x2;
public typedef float32_t3 f32x3;
public typedef float32_t4 f32x4;

public typedef float32_t2x2 f32x2x2;
public typedef float32_t2x3 f32x2x3;
public typedef float32_t3x2 f32x3x2;
public typedef float32_t3x3 f32x3x3;
public typedef float32_t3x4 f32x3x4;
public typedef float32_t4x4 f32x4x3;
public typedef float32_t4x4 f32x4x4;

public typedef float64_t f64;
public typedef float64_t2 f64x2;
public typedef float64_t3 f64x3;
public typedef float64_t4 f64x4;

__generic<T> public extension vector<T, 2> {
	public __init(T arr[2]) {
		return This(arr[0], arr[1]);
	}
}
__generic<T> public extension vector<T, 3> {
	public __init(T arr[3]) {
		return This(arr[0], arr[1], arr[2]);
	}
}
__generic<T> public extension vector<T, 4> {
	public __init(T arr[4]) {
		return This(arr[0], arr[1], arr[2], arr[3]);
	}
}

public interface Uniformity {
	static bool is_uniform();
}
public struct Uniform : Uniformity {
	[ForceInline]
	public static bool is_uniform() {
		return true;
	}
}
public struct NonUniform : Uniformity {
	[ForceInline]
	public static bool is_uniform() {
		return false;
	}
}

public interface TextureFormat {
	static const i32 Format;
}
public struct r32u : TextureFormat {
	public static const i32 Format = 37;
}
public struct r32f : TextureFormat {
	public static const i32 Format = 6;
}
public struct r64u : TextureFormat {
	public static const i32 Format = 40;
}
public struct rgba32f : TextureFormat {
	public static const i32 Format = 1;
}
