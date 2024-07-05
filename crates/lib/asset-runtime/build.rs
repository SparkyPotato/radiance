use radiance_shader_compiler::compile::ShaderBuilder;

fn main() {
	let mut b = ShaderBuilder::for_build();
	b.include("../asset-runtime").unwrap();
	b.build();
}
