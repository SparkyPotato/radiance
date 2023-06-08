struct Output {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

struct PushConstants {
	id: u32,
	aspect_ratio: f32,
}

struct Buffer {
	inner: array<mat4x4<f32>>,
}

@group(0) @binding(0)
var<storage> buffers: binding_array<Buffer>;
var<push_constant> pc: PushConstants;

@vertex
fn main(@builtin(vertex_index) id: u32) -> Output {
    var pos: vec2<f32>;
    var color: vec3<f32>;

    switch (id) {
        case 0u: {
            pos = vec2(0.0, 0.433); // Equilateral triangle spinning around the center.
            color = vec3(1.0, 0.0, 0.0);
        }
        case 1u: {
            pos = vec2(-0.5, -0.433);
            color = vec3(0.0, 1.0, 0.0);
        }
        case 2u: {
            pos = vec2(0.5, -0.433);
            color = vec3(0.0, 0.0, 1.0);
        }
        default: {}
    }

    let aspect = vec4(1.0 / pc.aspect_ratio, 1.0, 1.0, 1.0);
    let transform = buffers[pc.id].inner[0];
    let pos4 = vec4(pos, 0.0, 1.0);

    return Output(transform * pos4 * aspect, color);
}
