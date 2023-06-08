struct Output {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn main(@builtin(vertex_index) id: u32) -> Output {
    var pos: vec2<f32>;
    var color: vec3<f32>;

    switch (id) {
        case 0u: {
            pos = vec2(0.0, 0.5);
            color = vec3(1.0, 0.0, 0.0);
        }
        case 1u: {
            pos = vec2(-0.5, -0.5);
            color = vec3(0.0, 1.0, 0.0);
        }
        case 2u: {
            pos = vec2(0.5, -0.5);
            color = vec3(0.0, 0.0, 1.0);
        }
        default: {}
    }

    return Output(vec4(pos, 0.0, 1.0), color);
}
