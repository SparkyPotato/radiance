use std::{env, path::PathBuf};

fn main() {
	#[cfg(unix)]
	let builder;

	#[cfg(windows)]
	let mut builder;

	builder = bindgen::Builder::default().header("wrapper.h");

	#[cfg(windows)]
	let modify_builder = |nvtt_path: PathBuf| -> bindgen::Builder {
		let nvtt_path = nvtt_path.into_os_string().into_string().unwrap();
		let path = format!("-I{nvtt_path}\\include\\nvtt");
		println!("{}", &path);
		builder.clang_arg(path).clang_arg("-v")
	};

	if let Some(nvtt_path) = std::env::var_os("NVTT_PATH") {
		let nvtt_path = PathBuf::from(nvtt_path);
		cfg_if::cfg_if! {
			if #[cfg(unix)] {
				println!("cargo:rustc-link-search={}", nvtt_path.to_str().unwrap());
			}
			else if #[cfg(windows)] {
				println!("cargo:rustc-link-search={}", nvtt_path.to_str().unwrap());
				println!("cargo:rustc-link-search={}", nvtt_path.join(r"lib\x64-v142").to_str().unwrap());
				println!("cargo:rustc-link-search={}", nvtt_path.join(r"include\nvtt").to_str().unwrap());
				builder = modify_builder(nvtt_path);
			}
		}
	} else {
		#[cfg(windows)]
		{
			unsafe {
				let path_pw = windows::Win32::UI::Shell::SHGetKnownFolderPath(
					&windows::Win32::UI::Shell::FOLDERID_ProgramFiles,
					windows::Win32::UI::Shell::KNOWN_FOLDER_FLAG(0),
					windows::Win32::Foundation::HANDLE(0),
				)
				.expect("Failed to find Program Files");
				let pf = PathBuf::from(
					path_pw
						.to_string()
						.expect("Failed to make Program Files path into String"),
				);
				let nvtt_path = pf.join(r"NVIDIA Corporation\NVIDIA Texture Tools");

				println!("cargo:rustc-link-search={}", nvtt_path.to_str().unwrap());
				println!(
					"cargo:rustc-link-search={}",
					nvtt_path.join(r"lib\x64-v142").to_str().unwrap()
				);
				println!(
					"cargo:rustc-link-search={}",
					nvtt_path.join(r"include\nvtt").to_str().unwrap()
				);

				builder = modify_builder(nvtt_path);
			}
		}
	};

	cfg_if::cfg_if! {
		if #[cfg(windows)] {
			println!("cargo:rustc-link-lib=nvtt30205");
		} else if #[cfg(unix)] {
			println!("cargo:rustc-link-lib=nvtt");
		}
	}

	println!("cargo:rerun-if-changed=wrapper.h");

	let bindings = builder
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Enums
        .rustified_enum("NvttBoolean")
        .rustified_enum("NvttValueType")
        .rustified_enum("NvttChannelOrder")
        .rustified_enum("NvttFormat")
        .rustified_enum("NvttPixelType")
        .rustified_enum("NvttQuality")
        .rustified_enum("NvttWrapMode")
        .rustified_enum("NvttTextureType")
        .rustified_enum("NvttInputFormat")
        .rustified_enum("NvttMipmapFilter")
        .rustified_enum("NvttResizeFilter")
        .rustified_enum("NvttRoundMode")
        .rustified_enum("NvttAlphaMode")
        .rustified_enum("NvttError")
        .rustified_enum("NvttContainer")
        .rustified_enum("NvttNormalTransform")
        .rustified_enum("NvttToneMapper")
        .rustified_enum("NvttCubeLayout")
        .rustified_enum("EdgeFixup")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

	// Write the bindings to the $OUT_DIR/bindings.rs file.
	let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
	bindings
		.write_to_file(out_path.join("bindings.rs"))
		.expect("Couldn't write bindings!");
}
