import graph.types;
import common;

func srgb_gamma_from_linear(float3 rgb) -> f32x3 {
	let cutoff = rgb < 0.0031308;
	let lower = rgb * 12.92;
	let higher = 1.055 * pow(rgb, 1.0 / 2.4) - 0.055;
	return lerp(higher, lower, f32x3(cutoff));
}

// 0-1 sRGBA gamma  from  0-1 linear
func srgba_gamma_from_linear(float4 rgba) -> f32x4 {
	return f32x4(srgb_gamma_from_linear(rgba.rgb), rgba.a);
}

[shader("pixel")]
func main(input: VertexOutput) -> f32x4 {
	return Constants.image.sample(Constants.sampler, input.uv) * input.color;
}
