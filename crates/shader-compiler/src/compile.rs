use std::{error::Error, ffi::CString, path::PathBuf};

use slang::{
	Blob,
	CompileTarget,
	CompilerOptionEntry,
	CompilerOptionName,
	FloatingPointMode,
	GlobalSession,
	LineDirectiveMode,
	MatrixLayoutMode,
	Module,
	Session,
	SessionDescBuilder,
	TargetDescBuilder,
	TargetFlags,
};

use crate::c_str;

pub struct ShaderBuilder {
	session: GlobalSession,
	sesh: Session,
	source: PathBuf,
	cache: PathBuf,
}

unsafe impl Send for ShaderBuilder {}

impl ShaderBuilder {
	pub fn new(source: PathBuf, cache: PathBuf) -> Result<Self, Box<dyn Error>> {
		let sourcec = CString::new(source.as_os_str().to_owned().into_string().unwrap())?;
		let cachec = CString::new(cache.as_os_str().to_owned().into_string().unwrap())?;
		let search_paths = [sourcec.as_ptr() /* cachec.as_ptr() */];
		let session = GlobalSession::new()?;
		let sesh = session.create_session(
			SessionDescBuilder::default()
				.targets(&[TargetDescBuilder::default()
					.format(CompileTarget::SPIRV)
					.profile(session.find_profile("sm_6_7"))
					.flags(TargetFlags::GENERATE_SPIRV_DIRECTLY)
					.floating_point_mode(FloatingPointMode::FAST)
					.line_directive_mode(LineDirectiveMode::STANDARD)
					.force_glsl_scalar_buffer_layout(true)])
				.search_paths(&search_paths)
				.default_matrix_layout_mode(MatrixLayoutMode::COLUMN_MAJOR)
				.compiler_option_entries(&mut [
					CompilerOptionEntry::new(CompilerOptionName::LANGUAGE, c_str!("slang").as_ptr()),
					CompilerOptionEntry::new(CompilerOptionName::DEBUG_INFORMATION, 2),
					CompilerOptionEntry::new(CompilerOptionName::USE_UP_TO_DATE_BINARY_MODULE, 1),
					CompilerOptionEntry::new(CompilerOptionName::GLSL_FORCE_SCALAR_LAYOUT, 1),
				]),
		)?;
		Ok(Self {
			session,
			sesh,
			source,
			cache,
		})
	}

	fn load_raw(&mut self, name: &str) -> Result<Module, String> {
		let path = name.replace('.', "/");
		let mut module = self.sesh.load_module(&path).map_err(fmt_error)?;
		let path = self.cache.join(path).with_extension("slang-module");
		let _ = std::fs::create_dir_all(path.parent().unwrap());
		// let _ = module.write_to_file(path.as_os_str().to_str().unwrap());
		Ok(module)
	}

	pub fn load_module(&mut self, name: &str, entry: &str, spec: &[&str]) -> Result<Blob, String> {
		let mut module = self.load_raw(name)?;
		let mut entry = module
			.find_entry_point_by_name(entry)
			.map_err(|_| format!("could not find `{entry}` in `{name}`"))?;
		let prog = if spec.is_empty() {
			entry.link()
		} else {
			let mut others = Vec::with_capacity(spec.len());
			for n in spec {
				let m = self.load_raw(n)?;
				others.push((*m).clone());
			}
			others.push((*entry).clone());
			let mut speced = self.sesh.create_composite_component_type(&others).map_err(fmt_error)?;
			speced.link()
		}
		.map_err(fmt_error)?;
		prog.get_entry_point_code(0, 0).map_err(fmt_error)
	}
}

fn fmt_error(err: slang::Error) -> String {
	match err {
		slang::Error::Result(x) => format!("slang error: {x}"),
		slang::Error::Blob(d) => d.as_str().unwrap().to_string(),
	}
}
