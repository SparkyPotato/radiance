use radiance_shader_compiler::runtime::{shader, ShaderBlob};

pub mod asset;
pub mod debug;
pub mod mesh;
pub mod tonemap;

pub const SHADERS: ShaderBlob = shader!("radiance-passes");
