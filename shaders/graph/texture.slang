implementing graph;

[vk::binding(0, 0)]
__DynamicResource Textures[];

/// When targeting Vulkan, a combined-texture-sampler type (`isCombined==1`) translates to a `OpTypeSampledImage` type
/// in SPIR-V. For other targets, the combined-texture-sampler type is translated to a pair of a `Texture` and
/// `SamplerState`. `isShadow` is only applicable to combined-texture-sampler types and must be `0` for non-combined
/// texture types.
/// @internal
/// @category texture_types Texture types
// __magic_type(TextureType)
// __intrinsic_type($(kIROp_TextureType))
// struct _Texture<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let
// access:int, let isShadow:int, let isCombined:int, let format:int>;

public struct Tex<T : ITexelElement, S : __ITextureShape, U : Uniformity = Uniform> {
	u32 index;

	public __init() {
		this.index = 0;
	}

	__init(u32 index) {
		this.index = index;
	}

	public static Tex<T, S, U> from_index(u32 index) {
		return { index };
	}

	public u32 inner_index() {
		return this.index;
	}

	_Texture<T, S, 0, 0, 0, 0, 0, 0, 0> get() {
		if (U.is_uniform()) {
			return Textures[this.index].as<_Texture<T, S, 0, 0, 0, 0, 0, 0, 0>>();
		} else {
			return Textures[NonUniformResourceIndex(this.index)].as<_Texture<T, S, 0, 0, 0, 0, 0, 0, 0>>();
		}
	}

	public T load(vector<u32, S.dimensions> pixel, u32 mip = 0) {
		return this.get().Load(__makeVector(pixel, mip));
	}

	public T sample<SU : Uniformity>(Sampler<SU> sampler, vector<f32, S.dimensions> uv) {
		return this.get().Sample(sampler.get(), uv);
	}

	public T sample_mip<SU : Uniformity>(Sampler<SU> sampler, vector<f32, S.dimensions> uv, f32 mip) {
		return this.get().SampleLevel(sampler.get(), uv, mip);
	}

	public T sample_grad<SU : Uniformity>(Sampler<SU> sampler, vector<f32, S.dimensions> uv, f32 ddx, f32 ddy) {
		return this.get().SampleGrad(sampler.get(), uv, ddx, ddy);
	}
}

public struct OTex<T : ITexelElement, S : __ITextureShape, U : Uniformity = Uniform> {
	u32 index;

	public Optional<Tex<T, S, U>> get() {
		if (this.index == 0) {
			return none;
		} else {
			return Tex<T, S, U>.from_index(this.index);
		}
	}
}

public typealias D1 = __Shape1D;
public typealias D2 = __Shape2D;
public typealias D3 = __Shape3D;

__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias Tex1D = Tex<T, D1, U>;
__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias Tex2D = Tex<T, D2, U>;
__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias Tex3D = Tex<T, D3, U>;
__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias OTex1D = OTex<T, D1, U>;
__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias OTex2D = OTex<T, D2, U>;
__generic<T : ITexelElement, U : Uniformity = Uniform> public typealias OTex3D = OTex<T, D3, U>;

__generic<T : ITexelElement, U : Uniformity> public extension Tex1D<T, U> {
	public u32 size(u32 mip = 0) {
		u32 w, _;
		this.get().GetDimensions(mip, w, _);
		return w;
	}
}
__generic<T : ITexelElement, U : Uniformity> public extension Tex2D<T, U> {
	public u32x2 size(u32 mip = 0) {
		u32 w, h, _;
		this.get().GetDimensions(mip, w, h, _);
		return u32x2(w, h);
	}

	public u32x2 pixel_of_uv(f32x2 uv, u32 mip = 0) {
		f32x2 size = f32x2(this.size(mip));
		f32x2 xy = round(uv * size - 0.5f);
		return u32x2(xy);
	}
}
__generic<T : ITexelElement, U : Uniformity> public extension Tex3D<T, U> {
	public u32x3 size(u32 mip = 0) {
		u32 w, h, d, _;
		this.get().GetDimensions(mip, w, h, d, _);
		return u32x3(w, h, d);
	}
}

__generic<T : ITexelElement, S : __ITextureShape, U : Uniformity = Uniform> public extension Optional<Tex<T, S, U>> {
	public T load(vector<u32, S.dimensions> pixel, T default = T(T.Element(0)), u32 mip = 0) {
		if (this.hasValue) {
			return this.value.load(pixel, mip);
		} else {
			return default;
		}
	}

	public T sample<SU : Uniformity>(Sampler<SU> sampler, vector<f32, S.dimensions> uv, T default = T(T.Element(0))) {
		if (this.hasValue) {
			return this.value.sample(sampler, uv);
		} else {
			return default;
		}
	}

	public T sample_mip<SU : Uniformity>(Sampler<SU> sampler, vector<f32, S.dimensions> uv, f32 mip,
										 T default = T(T.Element(0))) {
		if (this.hasValue) {
			return this.value.sample_mip(sampler, uv, mip);
		} else {
			return default;
		}
	}
}

__generic<T : ITexelElement, U : Uniformity> public extension Optional<Tex1D<T, U>> {
	public u32 size(u32 mip = 0) {
		if (this.hasValue) {
			return this.value.size(mip);
		} else {
			return 0;
		}
	}
}
__generic<T : ITexelElement, U : Uniformity> public extension Optional<Tex2D<T, U>> {
	public u32x2 size(u32 mip = 0) {
		if (this.hasValue) {
			return this.value.size(mip);
		} else {
			return u32x2(0, 0);
		}
	}

	public u32x2 pixel_of_uv(f32x2 uv, u32 mip = 0) {
		if (this.hasValue) {
			return this.value.pixel_of_uv(uv, mip);
		} else {
			return u32x2(0, 0);
		}
	}
}
__generic<T : ITexelElement, U : Uniformity> public extension Optional<Tex3D<T, U>> {
	public u32x3 size(u32 mip = 0) {
		if (this.hasValue) {
			return this.value.size(mip);
		} else {
			return u32x3(0, 0, 0);
		}
	}
}

[vk::binding(1, 0)]
__DynamicResource STextures[];

public struct STex<T : ITexelElement, S : __ITextureShape, F : TextureFormat, U : Uniformity = Uniform> {
	u32 index;

	_Texture<T, S, 0, 0, 0, 1, 0, 0, F.Format> get() {
		if (U.is_uniform()) {
			return STextures[this.index].as<_Texture<T, S, 0, 0, 0, 1, 0, 0, F.Format>>();
		} else {
			return STextures[NonUniformResourceIndex(this.index)].as<_Texture<T, S, 0, 0, 0, 1, 0, 0, F.Format>>();
		}
	}

	public T load(vector<u32, S.dimensions> pixel, constexpr Optional<Scope> sc = none) {
		let img = this.get();
		if (let sc = sc) {
			return spirv_asm {
				OpExtension "SPV_KHR_vulkan_memory_model";
	            OpCapability VulkanMemoryModelKHR;
				%shrimpled:__sampledType(T) = OpImageRead $img $pixel NonPrivateTexel | MakeTexelVisible $sc;
				__truncate $$T result __sampledType(T) %shrimpled;
			};
		} else {
			return img[pixel];
		}
	}

	public void store(vector<u32, S.dimensions> pixel, T value, constexpr Optional<Scope> sc = none) {
		let img = this.get();
		if (let sc = sc) {
			spirv_asm {
				OpExtension "SPV_KHR_vulkan_memory_model";
	            OpCapability VulkanMemoryModelKHR;
				OpImageWrite $img $pixel __convertTexel(value) NonPrivateTexel | MakeTexelAvailable $sc;
			};
		} else {
			img[pixel] = value;
		}
	}
}
__generic<T : __BuiltinArithmeticType, S : __ITextureShape, F : TextureFormat, U : Uniformity> public extension
	STex<T, S, F, U>
	where T : ITexelElement {
	public T atomic_add(vector<u32, S.dimensions> pixel, T value, constexpr Scope sc = DEFAULT_SCOPE,
						constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
		return ::atomic_add(this.get()[pixel], value, sc, loc, se);
	}

	public T atomic_max(vector<u32, S.dimensions> pixel, T value, constexpr Scope sc = DEFAULT_SCOPE,
						constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
		return ::atomic_max(this.get()[pixel], value, sc, loc, se);
	}
}

__generic<T : ITexelElement, F : TextureFormat, U : Uniformity = Uniform> public typealias STex1D = STex<T, D1, F, U>;
__generic<T : ITexelElement, F : TextureFormat, U : Uniformity = Uniform> public typealias STex2D = STex<T, D2, F, U>;
__generic<T : ITexelElement, F : TextureFormat, U : Uniformity = Uniform> public typealias STex3D = STex<T, D3, F, U>;

__generic<T : ITexelElement, F : TextureFormat, U : Uniformity> public extension STex1D<T, F, U> {
	public u32 size() {
		u32 w;
		this.get().GetDimensions(w);
		return w;
	}
}
__generic<T : ITexelElement, F : TextureFormat, U : Uniformity> public extension STex2D<T, F, U> {
	public u32x2 size() {
		u32 w, h;
		this.get().GetDimensions(w, h);
		return u32x2(w, h);
	}

	public u32x2 pixel_of_uv(f32x2 uv) {
		f32x2 size = f32x2(this.size());
		f32x2 xy = round(uv * size - 0.5f);
		return u32x2(xy);
	}
}
__generic<T : ITexelElement, F : TextureFormat, U : Uniformity> public extension STex3D<T, F, U> {
	public u32x3 size() {
		u32 w, h, d;
		this.get().GetDimensions(w, h, d);
		return u32x3(w, h, d);
	}
}
