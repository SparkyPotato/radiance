struct Input {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@fragment
fn main(input: Input) -> @location(0) vec4<f32> {
    return vec4(input.color, 1.0);
}
