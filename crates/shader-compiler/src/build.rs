use crate::compile::ShaderModuleBuilder;

pub fn for_build_script(name: impl ToString) -> ShaderModuleBuilder {
	let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
	println!("cargo:rerun-if-changed={}/shader", root);
	ShaderModuleBuilder::new(name, root, std::env::var("OUT_DIR").unwrap()).unwrap()
}

pub fn export_env_var(builder: ShaderModuleBuilder) {
	let var_name = format!("{}_OUTPUT_PATH", builder.vfs.name);
	println!(
		"cargo:rustc-env={}={}",
		var_name,
		builder.vfs.output.join(format!("{}.spv", builder.vfs.name)).display()
	);
}

impl ShaderModuleBuilder {
	pub fn build(name: impl ToString) { for_build_script(name).run_build(); }

	pub fn run_build(mut self) {
		match self.compile_all() {
			Ok(()) => {},
			Err(e) => {
				for error in e {
					eprintln!("{error}");
				}
				panic!("Failed to compile shaders");
			},
		}
		match self.link() {
			Ok(()) => {},
			Err(e) => {
				eprintln!("{e}");
				panic!("Failed to link shaders");
			},
		}
		export_env_var(self)
	}
}
