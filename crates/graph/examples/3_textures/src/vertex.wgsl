struct Output {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PushConstants {
	image_id: u32,
	sampler_id: u32,
	aspect_ratio: f32,
}

struct Buffer {
	inner: array<mat4x4<f32>>,
}

var<push_constant> pc: PushConstants;

@vertex
fn main(@builtin(vertex_index) id: u32) -> Output {
    var pos: vec2<f32>;
    var uv: vec2<f32>;

    switch (id) {
        case 0u: {
            pos = vec2(-0.5, 0.5);
            uv = vec2(0.0, 0.0);
        }
        case 1u: {
            pos = vec2(0.5, 0.5);
            uv = vec2(1.0, 0.0);
        }
        case 2u: {
            pos = vec2(0.5, -0.5);
            uv = vec2(1.0, 1.0);
        }
        case 3u: {
            pos = vec2(-0.5, -0.5);
            uv = vec2(0.0, 1.0);
        }
        default: {}
    }

    let aspect = vec2(1.0 / pc.aspect_ratio, 1.0);
    let pos4 = vec4(pos * aspect, 0.0, 1.0);
    return Output(pos4, uv);
}
