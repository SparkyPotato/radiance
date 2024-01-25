use radiance_shader_compiler::runtime::{shader, ShaderBlob};

pub mod debug;
pub mod ground_truth;
pub mod mesh;

pub const SHADERS: ShaderBlob = shader!("radiance-passes");

