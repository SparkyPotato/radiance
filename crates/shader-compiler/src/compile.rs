use std::{error::Error, path::PathBuf};

use hassle_rs::{Dxc, DxcCompiler, DxcLibrary};
use slang::{
	Blob,
	CompileTarget,
	CompilerOptionEntry,
	CompilerOptionName,
	FileSystem,
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

struct CacheFs {
	source: PathBuf,
	cache: PathBuf,
}

impl FileSystem for CacheFs {
	fn load_file(&mut self, path: &str) -> Result<Blob, slang::Error> {
		let source = self.source.join(path);
		// let cache = self.cache.join(path);
		// if cache.exists() {
		// 	std::fs::read(cache)
		//} else {
		std::fs::read(source)
		//}
		.map(|x| x.into())
		.map_err(|x| slang::Error::Blob(x.to_string().into()))
	}
}

pub struct ShaderBuilder {
	_slang: GlobalSession,
	_dxc: Dxc,
	fs: CacheFs,
	sesh: Session,
	compiler: DxcCompiler,
	library: DxcLibrary,
}

unsafe impl Send for ShaderBuilder {}

impl ShaderBuilder {
	pub fn new(source: PathBuf, cache: PathBuf) -> Result<Self, Box<dyn Error>> {
		let fs = CacheFs {
			source: source.clone(),
			cache: cache.clone(),
		};
		let slang = GlobalSession::new()?;
		let sesh = slang.create_session(
			SessionDescBuilder::default()
				.targets(&[TargetDescBuilder::default()
					.format(CompileTarget::SPIRV)
					.profile(slang.find_profile("spirv_1_6"))
					.flags(TargetFlags::GENERATE_SPIRV_DIRECTLY)
					.floating_point_mode(FloatingPointMode::FAST)
					.line_directive_mode(LineDirectiveMode::STANDARD)
					.force_glsl_scalar_buffer_layout(true)])
				.file_system(CacheFs { source, cache })
				.search_paths(&[b".\0".as_ptr() as _])
				.default_matrix_layout_mode(MatrixLayoutMode::COLUMN_MAJOR)
				.compiler_option_entries(&mut [
					CompilerOptionEntry::new(CompilerOptionName::LANGUAGE, c_str!("slang").as_ptr()),
					CompilerOptionEntry::new(CompilerOptionName::DEBUG_INFORMATION, 2),
					CompilerOptionEntry::new(CompilerOptionName::USE_UP_TO_DATE_BINARY_MODULE, 1),
					CompilerOptionEntry::new(CompilerOptionName::GLSL_FORCE_SCALAR_LAYOUT, 1),
				]),
		)?;

		let dxc = Dxc::new(None)?;
		let compiler = dxc.create_compiler()?;
		let library = dxc.create_library()?;

		Ok(Self {
			_slang: slang,
			_dxc: dxc,
			fs,
			sesh,
			compiler,
			library,
		})
	}

	fn load_raw(&mut self, name: &str) -> Result<Module, String> {
		let path = name.replace('.', "/");
		let mut module = self.sesh.load_module(&path).map_err(fmt_error)?;
		let path = self.fs.cache.join(path).with_extension("slang-module");
		let _ = std::fs::create_dir_all(path.parent().unwrap());
		let _ = module.write_to_file(path.as_os_str().to_str().unwrap());
		Ok(module)
	}

	fn load_hlsl(&mut self, name: &str, entry: &str, spec: &[&str]) -> Result<Vec<u32>, String> {
		let defines: Vec<_> = spec
			.iter()
			.map(|&x| {
				if let Some((n, v)) = x.split_once('=') {
					(n, Some(v))
				} else {
					(x, None)
				}
			})
			.collect();
		let profile = match entry {
			"c" => "cs_6_8",
			_ => panic!("unknown type `{}`", entry),
		};
		let args: Vec<_> = [
			"-spirv",
			"-fspv-target-env=vulkan1.3",
			"-HV 2021",
			"-enable-16bit-types",
			"-ffinite-math-only",
			"-fvk-use-scalar-layout",
			"-fspv-reduce-load-size",
			"-Zi",
		]
		.into_iter()
		.map(|x| x.to_string())
		.chain(Some(format!("-T {}", profile)))
		.chain(Some(format!("-I {}", self.fs.source.display())))
		.collect();
		let args: Vec<_> = args.iter().map(|x| x.as_str()).collect();

		let shader = std::fs::read_to_string(name).map_err(|x| x.to_string())?;
		let blob = self
			.library
			.create_blob_with_encoding_from_str(&shader)
			.map_err(|x| x.to_string())?;
		match self
			.compiler
			.compile(&blob, name, "main", profile, &args, None, &defines)
		{
			Ok(result) => {
				let result_blob = result.get_result().unwrap();
				Ok(ash::util::read_spv(&mut std::io::Cursor::new(result_blob.as_slice()))
					.expect("failed to read spirv"))
			},
			Err((result, _)) => {
				let error_blob = result.get_error_buffer().unwrap();
				let e = self.library.get_blob_as_string(&error_blob.into()).unwrap();
				return Err(e);
			},
		}
	}

	pub fn load_module(&mut self, name: &str, entry: &str, spec: &[&str]) -> Result<Vec<u32>, String> {
		let hlsl = self.fs.source.join(name.replace('.', "/")).with_extension("hlsl");
		if hlsl.exists() {
			return self.load_hlsl(hlsl.as_os_str().to_str().unwrap(), entry, spec);
		}

		let mut module = self.load_raw(name)?;
		let mut sentry = module
			.find_entry_point_by_name(entry)
			.map_err(|_| format!("could not find `{entry}` in `{name}`"))?;
		let prog = if spec.is_empty() {
			sentry.link()
		} else {
			let mut others = Vec::with_capacity(spec.len());
			for n in spec {
				let m = self.load_raw(n)?;
				others.push((*m).clone());
			}
			others.push((*sentry).clone());
			let mut speced = self.sesh.create_composite_component_type(&others).map_err(fmt_error)?;
			speced.link()
		}
		.map_err(fmt_error)?;
		let blob = prog.get_entry_point_code(0, 0).map_err(fmt_error)?;

		let path = if spec.is_empty() {
			format!("{}.{}.spv", name, entry)
		} else {
			format!("{}.{}+{}.spv", name, entry, spec.join(","))
		};
		let f = self.fs.cache.join("spirv/");
		let _ = std::fs::create_dir_all(&f);
		let _ = std::fs::write(f.join(path), blob.as_slice());

		Ok(ash::util::read_spv(&mut std::io::Cursor::new(blob.as_slice())).expect("failed to read spirv"))
	}
}

fn fmt_error(err: slang::Error) -> String {
	match err {
		slang::Error::Result(x) => format!("slang error: {x}"),
		slang::Error::Blob(d) => d.as_str().unwrap().to_string(),
	}
}
