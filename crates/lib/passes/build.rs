use radiance_shader_compiler::compile::ShaderBuilder;

fn main() {
	let mut builder = ShaderBuilder::for_build();
	builder.include("../core").unwrap();
	builder.build();
}
