//! Shader compiler and runtime loader for radiance.
//!
//! Each crate can have its own shader module, which is a collection of shaders located in the `shaders/` directory
//! alongside `Cargo.toml`.
//!
//! To build shaders, `slangc` must be present in the path.

#[cfg(feature = "build-script")]
pub mod build;

#[cfg(feature = "compile")]
pub mod compile;

#[cfg(feature = "runtime")]
pub mod runtime;
