import graph;
import graph.util;
import graph.util.color;

struct PushConstants {
	Tex2D<f32x4> input;
	f32* exp;
};

[vk::push_constant]
PushConstants Constants;

// Don't rember where I stole this from, but:
// The code in this file was originally written by Stephen Hill (@self_shadow), who deserves all
// credit for coming up with this fit and implementing it. Buy him a beer next time you see him. :)

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const f32x3x3 ACESInputMat = {
	{ 0.59719, 0.35458, 0.04823 }, { 0.07600, 0.90834, 0.01566 }, { 0.02840, 0.13383, 0.83777 }
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const f32x3x3 ACESOutputMat = {
	{ 1.60475, -0.53108, -0.07367 }, { -0.10208, 1.10813, -0.00605 }, { -0.00327, -0.07276, 1.07602 }
};

f32x3 RRTAndODTFit(f32x3 v) {
	let a = v * (v + 0.0245786f) - 0.000090537f;
	let b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
	return a / b;
}

f32x3 ACESFitted(f32x3 color) {
	var yxy = rgb_to_Yxy(color);
	let l_max = 1.2f * exp2(*Constants.exp);
	yxy.x /= l_max + 0.0001f;
	color = Yxy_to_rgb(yxy);

	color = mul(ACESInputMat, color);

	// Apply RRT and ODT
	color = RRTAndODTFit(color);

	color = mul(ACESOutputMat, color);

	// Clamp to [0, 1]
	color = saturate(color);

	return color;
}

[shader("pixel")]
f32x4 main(ScreenOutput input) : SV_Target0 {
	let pixel = Constants.input.pixel_of_uv(input.uv);
	let color = Constants.input.load(pixel).xyz;
	return float4(ACESFitted(color), 1.f);
}
