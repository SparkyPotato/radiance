implementing graph;

[vk::binding(0, 0)]
__DynamicResource Textures[];

public struct Tex<T, Shape : __ITextureShape, U : Uniformity = Uniform> {
	u32 index;

	__TextureImpl<T, Shape, 0, 0, 0, 0, 0, 0, 0> get() {
		return Textures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 0, 0, 0, 0>>();
	}

	public T sample<S : Uniformity>(Sampler<S> sampler, vector<f32, Shape.dimensions> uv) {
		return this.get().Sample(sampler.get(), uv);
	}

	public T sample_mip<S : Uniformity>(Sampler<S> sampler, vector<f32, Shape.dimensions> uv, f32 mip) {
		return this.get().SampleLevel(sampler.get(), uv, mip);
	}
}

public typealias D1 = __Shape1D;
public typealias D2 = __Shape2D;
public typealias D3 = __Shape3D;

__generic<T, U : Uniformity> public extension Tex<T, D1, U> {
	public T load(u32 pixel, u32 mip = 0) {
		return this.get().Load(i32x2(pixel, mip));
	}

	public u32 size(u32 mip = 0) {
		u32 w, _;
		this.get().GetDimensions(mip, w, _);
		return w;
	}
}
__generic<T, U : Uniformity> public extension Tex<T, D2, U> {
	public T load(u32x2 pixel, u32 mip = 0) {
		return this.get().Load(i32x3(pixel, mip));
	}

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
__generic<T, U : Uniformity> public extension Tex<T, D3, U> {
	public T load(u32x3 pixel, u32 mip = 0) {
		return this.get().Load(i32x4(pixel, mip));
	}

	public u32x3 size(u32 mip = 0) {
		u32 w, h, d, _;
		this.get().GetDimensions(mip, w, h, d, _);
		return u32x3(w, h, d);
	}
}

[vk::binding(1, 0)]
__DynamicResource STextures[];
[vk::binding(1, 0)]
coherent __DynamicResource CTextures[];

public struct STex<T : __BuiltinArithmeticType, Shape : __ITextureShape, C : Coherency = Incoherent,
				   U : Uniformity = Uniform> {
	u32 index;

	__TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0> get() {
		if (C.is_coherent())
			return CTextures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0>>();

		else
			return STextures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0>>();
	}

	public T load(vector<u32, Shape.dimensions> pixel) {
		if (C.is_coherent()) {
			let img = this.get();
			u32 sc = DEFAULT_SCOPE;
			return spirv_asm {
				OpCapability VulkanMemoryModel;
				result:$$T = OpImageRead $img $pixel NonPrivateTexel | MakeTexelVisible $sc;
			};
		} else {
			return this.get().Load(pixel);
		}
	}

	public void store(vector<u32, Shape.dimensions> pixel, T value) {
		if (C.is_coherent()) {
			let img = this.get();
			u32 sc = DEFAULT_SCOPE;
			spirv_asm {
				OpCapability VulkanMemoryModel;
				OpImageWrite $img $pixel $value NonPrivateTexel | MakeTexelAvailable $sc;
			};
		} else {
			this.get()[pixel] = value;
		}
	}

	public T atomic_add(vector<u32, Shape.dimensions> pixel, T value, constexpr Scope sc = DEFAULT_SCOPE,
						constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
		return ::atomic_add(this.get()[pixel], value, sc, loc, se);
	}

	public T atomic_max(vector<u32, Shape.dimensions> pixel, T value, constexpr Scope sc = DEFAULT_SCOPE,
						constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
		return ::atomic_max(this.get()[pixel], value, sc, loc, se);
	}
}

__generic<T : __BuiltinArithmeticType, C : Coherency, U : Uniformity> public extension STex<T, D1, C, U> {
	public u32 size() {
		u32 w;
		this.get().GetDimensions(w);
		return w;
	}
}
__generic<T : __BuiltinArithmeticType, C : Coherency, U : Uniformity> public extension STex<T, D2, C, U> {
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
__generic<T : __BuiltinArithmeticType, C : Coherency, U : Uniformity> public extension STex<T, D3, C, U> {
	public u32x3 size() {
		u32 w, h, d;
		this.get().GetDimensions(w, h, d);
		return u32x3(w, h, d);
	}
}
