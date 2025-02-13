use std::{collections::HashMap, ops::Deref, slice};

use slang::{
    Blob, CompileTarget, FileSystem, GlobalSession, IntoError, SessionDescBuilder,
    TargetDescBuilder, TargetFlags,
};

struct MyFileSystem(HashMap<String, String>);

impl MyFileSystem {
    fn new() -> Self {
        let mut m = HashMap::new();

        m.insert(
            "utils.slang".to_owned(),
            r#"
func get_increment() -> uint {
    return 1;
}
"#
            .to_owned(),
        );

        m.insert(
            "example.slang".to_owned(),
            r#"
import utils;

struct MyValue {
    uint value;
}

[[vk::push_constant]] struct PushConstants {
    MyValue* my_ptr;
} constants;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    InterlockedAdd(constants.my_ptr.value, get_increment());
}"#
            .to_owned(),
        );

        Self(m)
    }
}

impl FileSystem for MyFileSystem {
    fn load_file(&mut self, path: &str) -> Result<Blob, slang::Error> {
        self.0
            .get(path)
            .cloned()
            .map(|source| Blob::from(source))
            .into_error()
    }
}

fn main() {
    let global_session = GlobalSession::new().unwrap();

    let target_desc = TargetDescBuilder::default()
        .format(CompileTarget::SPIRV)
        .profile(global_session.find_profile("spirv_1_4"))
        .flags(TargetFlags::GENERATE_SPIRV_DIRECTLY)
        .force_glsl_scalar_buffer_layout(true);

    let session_desc = SessionDescBuilder::default()
        .targets(slice::from_ref(&target_desc))
        .file_system(MyFileSystem::new());

    let mut session = global_session.create_session(session_desc).unwrap();
    let mut module = session.load_module("example").unwrap();

    let entry_point = module.find_entry_point_by_name("main").unwrap();

    let mut program = session
        .create_composite_component_type(&[module.deref().clone(), entry_point.deref().clone()])
        .unwrap();
    let linked_program = program.link().unwrap();
    let code = linked_program.get_entry_point_code(0, 0).unwrap();
    println!("{:?}", code.as_slice());
}
