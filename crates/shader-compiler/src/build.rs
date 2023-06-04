//! Utilities for compiling shaders in a build script.

use std::path::Path;

use crate::compile::ShaderBuilder;

impl ShaderBuilder {
	/// Create a new shader module builder for the build script. The name of the shader module is the same name as the
	/// crate.
	pub fn for_build() -> Self {
		let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
		println!("cargo:rerun-if-changed={}/shaders", root);

		let mut builder = ShaderBuilder::new().unwrap();
		builder
			.target(root.as_ref(), std::env::var("OUT_DIR").unwrap().as_ref())
			.unwrap();

		builder
	}

	/// Compile all shaders in the shader module.
	///
	/// The environment variable `name_OUTPUT_PATH` will be set to the path of the compiled shader module.
	pub fn build(mut self) {
		let link = match self.compile_all() {
			Ok(x) => x,
			Err(e) => {
				for error in e {
					eprintln!("{error}");
				}
				panic!("Failed to compile shaders");
			},
		};

		if link {
			match self.link() {
				Ok(()) => {},
				Err(e) => {
					for error in e {
						eprintln!("{error}");
					}
					panic!("Failed to link shaders");
				},
			}
		}

		for (name, _, out) in self.vfs.compilable_modules() {
			let var_name = format!("{}_OUTPUT_PATH", name);
			println!(
				"cargo:rustc-env={}={}",
				var_name,
				out.parent().unwrap().join(format!("{}.spv", name)).display()
			);
		}

		self.write_deps(&Path::new(&std::env::var("OUT_DIR").unwrap()).join("dependencies.json"))
			.unwrap();
	}
}
