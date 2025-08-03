use std::fs;

use bindgen::Formatter;

fn generate_bindings() {
    fs::create_dir_all("gen").unwrap();

    bindgen::Builder::default()
        .header("vendor/slang/include/slang.h")
        .clang_arg("-I./vendor/include/slang")
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .allowlist_function("slang_.*")
        .allowlist_type("slang.*")
        .allowlist_type("FileSystemContentsCallBack")
        .allowlist_type("PathKind")
        .allowlist_var("SLANG_.*")
        .with_codegen_config(
            bindgen::CodegenConfig::FUNCTIONS
                | bindgen::CodegenConfig::TYPES
                | bindgen::CodegenConfig::VARS,
        )
        .layout_tests(false)
        .vtable_generation(true)
        .derive_copy(true)
        .formatter(Formatter::Rustfmt)
        .generate()
        .expect("Failed to generate bindings")
        .write_to_file("gen/bindings.rs")
        .expect("Failed to write bindings to file");
}

fn main() {
    generate_bindings();

    let path = cmake::Config::new("vendor/slang")
        .profile("RelWithDebInfo")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "0")
        .define("SLANG_LIB_TYPE", "STATIC")
        .define("SLANG_ENABLE_GFX", "0")
        .define("SLANG_ENABLE_SLANGC", "0")
        .define("SLANG_ENABLE_SLANGD", "0")
        .define("SLANG_ENABLE_SLANG_GLSLANG", "0")
        .define("SLANG_ENABLE_SLANGRT", "0")
        .define("SLANG_ENABLE_PREBUILT_BINARIES", "0")
        .define("SLANG_ENABLE_TESTS", "0")
        .define("SLANG_ENABLE_EXAMPLES", "0")
        .define("SLANG_SLANG_LLVM_FLAVOR", "DISABLE")
        .define("SLANG_ENABLE_RELEASE_LTO", "0")
        .define("LZ4_BUILD_CLI", "0")
        .build_target("slang")
        .generator("Ninja")
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        path.join("build/RelWithDebInfo/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        path.join("build/external/miniz").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        path.join("build/external/lz4/build/cmake").display()
    );
    println!("cargo:rustc-link-lib=slang");
    println!("cargo:rustc-link-lib=compiler-core");
    println!("cargo:rustc-link-lib=core");
    println!("cargo:rustc-link-lib=miniz");
    println!("cargo:rustc-link-lib=lz4");
}
