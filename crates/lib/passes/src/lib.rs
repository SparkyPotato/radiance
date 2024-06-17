#![feature(f16)]

use radiance_shader_compiler::runtime::{shader, ShaderBlob};

pub mod cpu_path;
pub mod debug;
pub mod ground_truth;
pub mod mesh;
pub mod tonemap;

pub const SHADERS: ShaderBlob = shader!("radiance-passes");
