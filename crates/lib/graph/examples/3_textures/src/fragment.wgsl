struct Input {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PushConstants {
	image_id: u32,
	sampler_id: u32,
	aspect_ratio: f32,
}

@group(0) @binding(1)
var images: binding_array<texture_2d<f32>>;
@group(0) @binding(3)
var samplers: binding_array<sampler>;

var<push_constant> pc: PushConstants;

@fragment
fn main(input: Input) -> @location(0) vec4<f32> {
    return textureSample(images[pc.image_id], samplers[pc.sampler_id], input.uv);
}
