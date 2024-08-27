use std::path::{Path, PathBuf};

use proc_macro::TokenStream;
use quote::quote;
use walkdir::WalkDir;

#[proc_macro]
pub fn shader(stream: TokenStream) -> TokenStream {
	let mut iter = stream.into_iter();
	let name = iter.next().expect("expected shader module name");
	assert!(iter.next().is_none(), "expected only one argument");
	let name = name.to_string()[1..name.to_string().len() - 1].to_string();
	let path = std::env::var(format!("{}_OUTPUT_PATH", name)).expect("shader module not found");
	let path = Path::new(&path);

	let modules = WalkDir::new(path)
		.into_iter()
		.filter_map(|x| x.ok())
		.filter(|x| x.path().is_file() && matches!(x.path().extension().and_then(|x| x.to_str()), Some("spv")))
		.map(|spirv| {
			let relative = spirv.path().strip_prefix(path).unwrap();
			let relative = relative.with_extension("").with_extension("");
			let mut name = PathBuf::from(&name);
			name.push(relative);
			let name = name.to_str().unwrap().replace("\\", "/");
			let path = spirv.path().to_str().unwrap();
			quote! {
				(#name, ::std::include_bytes!(#path))
			}
		});
	let source_path = std::env::var(format!("{}_SOURCE_PATH", name)).expect("shader module not found");
	let path = path.to_str().unwrap();

	(quote! {
		::radiance_shader_compiler::runtime::ShaderBlob::new(
			#source_path,
			#path,
			&[#(#modules),*]
		)
	})
	.into()
}
