use std::{ops::Deref, slice};

use slang::{
    Blob, CompileTarget, GlobalSession, SessionDescBuilder, TargetDescBuilder, TargetFlags,
};

fn main() {
    let global_session = GlobalSession::new().unwrap();

    let target_desc = TargetDescBuilder::default()
        .format(CompileTarget::SPIRV)
        .profile(global_session.find_profile("spirv_1_4"))
        .flags(TargetFlags::GENERATE_SPIRV_DIRECTLY)
        .force_glsl_scalar_buffer_layout(true);

    let session_desc = SessionDescBuilder::default().targets(slice::from_ref(&target_desc));

    let mut session = global_session.create_session(session_desc).unwrap();

    let blob = Blob::from(
        r#"
struct MyValue {
    uint value;
}

[[vk::push_constant]] struct PushConstants {
    MyValue* my_ptr;
} constants;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    InterlockedAdd(constants.my_ptr.value, 5);
}"#,
    );

    let mut module = session
        .load_module_from_source("example", "example.slang", &blob)
        .unwrap();

    let entry_point = module.find_entry_point_by_name("main").unwrap();

    let mut program = session
        .create_composite_component_type(&[module.deref().clone(), entry_point.deref().clone()])
        .unwrap();
    let linked_program = program.link().unwrap();
    let code = linked_program.get_entry_point_code(0, 0).unwrap();
    println!("{:?}", code.as_slice());
}
