//! Utilities for compiling shaders in a build script.

use std::path::Path;

use crate::compile::ShaderBuilder;

impl ShaderBuilder {
	/// Create a new shader module builder for the build script. The name of the shader module is the same name as the
	/// crate.
	pub fn for_build() -> Self {
		println!("cargo:rerun-if-changed=../..");

		let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
		let out = std::env::var("OUT_DIR").unwrap();

		let mut builder = ShaderBuilder::new().unwrap();
		let _ = builder.deps(&Path::new(&out).join("dependencies.json"));
		builder.target(root.as_ref(), out.as_ref()).unwrap();

		builder
	}

	/// Compile all shaders in the shader module.
	///
	/// The environment variable `name_OUTPUT_PATH` will be set to the path of the compiled shader module.
	pub fn build(mut self) {
		match self.compile_all() {
			Ok(x) => x,
			Err(e) => {
				for error in e {
					eprintln!("{error}");
				}
				panic!("Failed to compile shaders");
			},
		}

		for (name, source, out) in self.vfs.compilable_modules() {
			println!("cargo:rustc-env={}={}", format!("{}_OUTPUT_PATH", name), out.display());
			println!(
				"cargo:rustc-env={}={}",
				format!("{}_SOURCE_PATH", name),
				source.parent().unwrap().display()
			);
		}

		self.write_deps(&Path::new(&std::env::var("OUT_DIR").unwrap()).join("dependencies.json"))
			.unwrap();
	}
}
