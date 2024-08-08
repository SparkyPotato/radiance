module texture;

import types;
import sampler;

[vk::binding(0, 0)] __DynamicResource Textures[];

public struct Tex<T, Shape: __ITextureShape, U : Uniformity = Uniform> {
	u32 index;

	func get() -> __TextureImpl<T, Shape, 0, 0, 0, 0, 0, 0, 0> {
		return Textures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 0, 0, 0, 0>>();
	}

	public func sample<S : Uniformity>(Sampler<S> sampler, vector<f32, Shape.dimensions> uv) -> T {
		return this.get().Sample(sampler.get(), uv);
	}

	public func sample_mip<S : Uniformity>(Sampler<S> sampler, vector<f32, Shape.dimensions> uv, f32 mip) -> T {
		return this.get().SampleLevel(sampler.get(), uv, mip);
	}
}

public typealias D1 = __Shape1D;
public typealias D2 = __Shape2D;
public typealias D3 = __Shape3D;

__generic<T, U : Uniformity>
public extension Tex<T, D1, U> {
	public func load(u32 pixel, u32 mip = 0) -> T {
		return this.get().Load(i32x2(pixel, mip));
	}

	public func size(u32 mip = 0) -> u32 {
		u32 w, _;
		this.get().GetDimensions(mip, w, _);
		return w;
	}
}
__generic<T, U : Uniformity>
public extension Tex<T, D2, U> {
	public func load(u32x2 pixel, u32 mip = 0) -> T {
		return this.get().Load(i32x3(pixel, mip));
	}

	public func size(u32 mip = 0) -> u32x2 {
		u32 w, h, _;
		this.get().GetDimensions(mip, w, h, _);
		return u32x2(w, h);
	}

	public func pixel_of_uv(f32x2 uv, u32 mip = 0) -> u32x2 {
		f32x2 size = f32x2(this.size(mip));
		f32x2 xy = round(uv * size - 0.5f);
		return u32x2(xy);
	}
}
__generic<T, U : Uniformity>
public extension Tex<T, D3, U> {
	public func load(u32x3 pixel, u32 mip = 0) -> T {
		return this.get().Load(i32x4(pixel, mip));
	}

	public func size(u32 mip = 0) -> u32x3 {
		u32 w, h, d, _;
		this.get().GetDimensions(mip, w, h, d, _);
		return u32x3(w, h, d);
	}
}

[vk::binding(1, 0)] __DynamicResource STextures[];

public struct STex<T, Shape: __ITextureShape, C : Coherency = Incoherent, U : Uniformity = Uniform> {
	u32 index;

	func get() -> __TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0> {
		if (C.is_coherent()) {
			return STextures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0>>();
		} else {
			return STextures[U.wrap_index(this.index)].as<__TextureImpl<T, Shape, 0, 0, 0, 1, 0, 0, 0>>();
		}
	}

	public func load(vector<u32, Shape.dimensions> pixel) -> T {
		return this.get().Load(pixel);
	}

	public func store(vector<u32, Shape.dimensions> pixel, T value) {
		this.get()[pixel] = value;
	}
}

__generic<T, C : Coherency, U : Uniformity>
public extension STex<T, D1, C, U> {
	public func size(u32 mip = 0) -> u32 {
		u32 w, _;
		this.get().GetDimensions(mip, w, _);
		return w;
	}
}
__generic<T, C : Coherency, U : Uniformity>
public extension STex<T, D2, C, U> {
	public func size(u32 mip = 0) -> u32x2 {
		u32 w, h, _;
		this.get().GetDimensions(mip, w, h, _);
		return u32x2(w, h);
	}
}
__generic<T, C : Coherency, U : Uniformity>
public extension STex<T, D3, C, U> {
	public func size(u32 mip = 0) -> u32x3 {
		u32 w, h, d, _;
		this.get().GetDimensions(mip, w, h, d, _);
		return u32x3(w, h, d);
	}
}
