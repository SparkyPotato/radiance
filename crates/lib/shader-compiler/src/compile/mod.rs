use std::{
	borrow::Borrow,
	error::Error,
	ffi::OsStr,
	fs::File,
	io::{BufReader, Read},
	path::{Path, PathBuf},
	process::{Command, Stdio},
};

use hassle_rs::{Dxc, DxcCompiler3, DxcIncludeHandler, DxcLibrary};
use rspirv::{binary::Assemble, dr::Operand};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::compile::vfs::{VirtualFileSystem, VirtualPath, VirtualPathBuf};

pub mod vfs;

pub struct ShaderBuilder {
	pub(crate) vfs: VirtualFileSystem,
	dependencies: DependencyInfo,
	compiler: DxcCompiler3,
	library: DxcLibrary,
	_dxc: Dxc,
}

impl ShaderBuilder {
	pub fn new() -> Result<Self, Box<dyn Error>> {
		let dxc = Dxc::new(None)?;
		let compiler = dxc.create_compiler3()?;
		let library = dxc.create_library()?;

		Ok(Self {
			vfs: VirtualFileSystem::new(),
			dependencies: DependencyInfo::default(),
			compiler,
			library,
			_dxc: dxc,
		})
	}

	pub fn deps(&mut self, file: &Path) -> Result<(), Box<dyn Error>> {
		let mut file = BufReader::new(File::open(file)?);
		let deps: DependencyInfo = serde_json::from_reader(&mut file)?;
		self.dependencies.merge(deps);
		Ok(())
	}

	/// Include files from another shader module.
	pub fn include(&mut self, source_path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
		self.vfs.add_root(source_path.as_ref(), None)
	}

	/// Add a new target for building.
	pub fn target(&mut self, source_path: &Path, output_path: &Path) -> Result<(), Box<dyn Error>> {
		self.vfs.add_root(source_path, Some(output_path))
	}

	/// Compile a physical file.
	pub fn compile_file_physical(&mut self, file: &Path) -> Result<(), String> {
		let ty = ShaderType::new(file)?;
		let virtual_path = self.vfs.unresolve_source(file).unwrap();

		if let Some(ty) = ty {
			let args: Vec<_> = [
				"-spirv",
				"-fspv-target-env=vulkan1.3",
				"-HV 2021",
				"-enable-16bit-types",
			]
			.into_iter()
			.map(|x| x.to_string())
			.chain(Some(format!("-T {}", ty.target_profile())))
			.collect();
			let args: Vec<_> = args.iter().map(|x| x.as_str()).collect();
			let mut handler = IncludeHandler::new(&self.vfs, &mut self.dependencies, &virtual_path);

			let shader = std::fs::read_to_string(file).map_err(|x| x.to_string())?;
			let bytecode: Vec<u8> = match self.compiler.compile(&shader, &args, Some(&mut handler)) {
				Ok(result) => {
					let result_blob = result.get_result().unwrap();
					result_blob.to_vec()
				},
				Err((result, _)) => {
					let error_blob = result.get_error_buffer().unwrap();
					let e = self.library.get_blob_as_string(&error_blob.into()).unwrap();
					return Err(e);
				},
			};

			let output = self.vfs.resolve_output(&virtual_path).unwrap();
			std::fs::create_dir_all(output.parent().unwrap()).unwrap();

			let mut spirv = rspirv::dr::load_bytes(bytecode).unwrap();
			if spirv.entry_points.len() != 1 {
				return Err(format!("Shader `{}` must have exactly one entry point", file.display()));
			}

			let mut x = PathBuf::from(format!("{}", virtual_path.display()));
			x.set_extension("");
			let name = x.to_str().unwrap().to_string();
			let entry = spirv.entry_points.iter_mut().next().unwrap();
			let n = entry
				.operands
				.iter_mut()
				.find_map(|x| match x {
					Operand::LiteralString(s) => Some(s),
					_ => None,
				})
				.unwrap();
			*n = name;
			let bytecode = spirv.assemble();
			std::fs::write(output, unsafe {
				std::slice::from_raw_parts(
					bytecode.as_ptr() as *const u8,
					bytecode.len() * std::mem::size_of::<u32>(),
				)
			})
			.unwrap()
		}

		Ok(())
	}

	/// Compile all shaders in all modules.
	///
	/// Returns `true` if any shaders were compiled.
	pub fn compile_all(&mut self) -> Result<bool, Vec<String>> {
		let mut errors = Vec::new();
		let mut compile_queue = Vec::new();

		for (_, module_root, _) in self.vfs.compilable_modules() {
			for file in WalkDir::new(module_root)
				.into_iter()
				.filter_map(|e| e.ok())
				.filter(|e| e.path().is_file())
			{
				let path = file.path();
				let virtual_path = self.vfs.unresolve_source(path).unwrap();
				let output_path = self.vfs.resolve_output(&virtual_path).unwrap();

				if let Ok(meta) = std::fs::metadata(output_path) {
					if meta.modified().unwrap() < file.metadata().unwrap().modified().unwrap() {
						compile_queue.push(path.to_path_buf());
						compile_queue.extend(
							self.dependencies
								.on(&virtual_path)
								.map(|x| self.vfs.resolve_source(x).unwrap()),
						);
					}
				} else {
					compile_queue.push(path.to_path_buf());
				}
			}
		}

		let compiled = !compile_queue.is_empty();
		for file in compile_queue {
			eprintln!("Compiling {}", file.display());

			if let Err(e) = self.compile_file_physical(&file) {
				errors.push(e)
			}
		}

		if errors.is_empty() {
			Ok(compiled)
		} else {
			Err(errors)
		}
	}

	// Link all modules into a single SPIRV modules.
	pub fn link(&mut self) -> Result<(), Vec<String>> {
		let mut errors = Vec::new();

		for (name, _, module_out) in self.vfs.compilable_modules() {
			let compile = || {
				let files: Vec<_> = WalkDir::new(module_out)
					.into_iter()
					.filter_map(|e| e.ok())
					.filter_map(|e| match e.path().extension().and_then(|x| x.to_str()) {
						Some("spv") => Some(e.path().as_os_str().to_owned()),
						_ => None,
					})
					.collect();
				let out_path = module_out.parent().unwrap().join(format!("{}.spv", name));
				let ret = Command::new("spirv-link")
					.stdout(Stdio::null())
					.stderr(Stdio::piped())
					.args(
						[
							OsStr::new("--target-env"),
							OsStr::new("vulkan1.3"),
							OsStr::new("-o"),
							out_path.as_os_str(),
						]
						.into_iter()
						.chain(files.iter().map(|x| x.as_os_str())),
					)
					.spawn()
					.map_err(|x| x.to_string())?;

				let mut stderr = String::new();
				ret.stderr
					.unwrap()
					.read_to_string(&mut stderr)
					.map_err(|x| x.to_string())?;
				if !stderr.is_empty() {
					Err(stderr)
				} else {
					Ok(())
				}
			};

			if let Err(e) = compile() {
				errors.push(e);
			}
		}

		if errors.is_empty() {
			Ok(())
		} else {
			Err(errors)
		}
	}

	pub fn write_deps(&self, path: &Path) -> Result<(), Box<dyn Error>> {
		let out = serde_json::to_string(&self.dependencies)?;
		std::fs::write(path, out)?;

		Ok(())
	}
}

struct IncludeHandler<'a> {
	vfs: &'a VirtualFileSystem,
	deps: &'a mut DependencyInfo,
	curr: &'a VirtualPath,
}

impl<'a> IncludeHandler<'a> {
	fn new(vfs: &'a VirtualFileSystem, deps: &'a mut DependencyInfo, curr: &'a VirtualPath) -> Self {
		Self { vfs, deps, curr }
	}

	fn load(&mut self, filename: &Path) -> Option<String> {
		match std::fs::read_to_string(filename) {
			Ok(source) => {
				self.deps.add(self.curr, self.vfs.unresolve_source(filename)?);
				Some(source)
			},
			Err(_) => None,
		}
	}
}

impl DxcIncludeHandler for IncludeHandler<'_> {
	fn load_source(&mut self, filename: String) -> Option<String> {
		let path = Path::new(&filename);
		let mut comp = path.components();
		comp.next().unwrap();
		let path = comp.as_path();
		let us = self.vfs.resolve_source(self.curr)?;
		let curr_dir = us.parent()?;
		let curr_module = self.curr.get_module();

		// Check in current directory
		match self.load(&curr_dir.join(path)) {
			Some(source) => Some(source),
			// Current module root
			None => match self.load(&self.vfs.get_root(curr_module)?.join(path)) {
				Some(source) => Some(source),
				None => self.load(&self.vfs.resolve_source(VirtualPath::new(&path))?),
			},
		}
	}
}

#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
struct DependencyInfo {
	inner: FxHashMap<VirtualPathBuf, FxHashSet<VirtualPathBuf>>,
}

impl DependencyInfo {
	pub fn merge(&mut self, other: Self) {
		for (k, v) in other.inner {
			self.inner.entry(k).or_default().extend(v);
		}
	}

	pub fn add(&mut self, file: impl Into<VirtualPathBuf>, depends_on: impl Into<VirtualPathBuf>) {
		let on = depends_on.into();
		let file = file.into();
		self.inner.entry(on).or_default().insert(file);
	}

	pub fn on(&self, file: impl AsRef<VirtualPath>) -> impl Iterator<Item = &VirtualPath> + '_ {
		self.inner
			.get(file.as_ref())
			.into_iter()
			.flat_map(|x| x.iter())
			.map(|x| x.borrow())
	}
}

#[derive(Copy, Clone)]
enum ShaderType {
	Pixel,
	Vertex,
	Geometry,
	Hull,
	Domain,
	Compute,
	Mesh,
	Amplification,
}

impl ShaderType {
	fn new(path: &Path) -> Result<Option<Self>, String> {
		let err = || {
			format!(
				"Shader file name must follow format <name>.<type>.hlsl (is `{}`)",
				path.display()
			)
		};

		let mut dots = path.file_name().unwrap().to_str().unwrap().split('.');
		let _ = dots.next().ok_or_else(err)?;
		let ty = dots.next().ok_or_else(err)?;

		let ty = match ty {
			"p" => Some(ShaderType::Pixel),
			"v" => Some(ShaderType::Vertex),
			"g" => Some(ShaderType::Geometry),
			"h" => Some(ShaderType::Hull),
			"d" => Some(ShaderType::Domain),
			"c" => Some(ShaderType::Compute),
			"m" => Some(ShaderType::Mesh),
			"a" => Some(ShaderType::Amplification),
			"hlsl" => return Err(err()),
			_ => None,
		};

		Ok(ty)
	}

	fn target_profile(self) -> &'static str {
		match self {
			ShaderType::Pixel => "ps_6_7",
			ShaderType::Vertex => "vs_6_7",
			ShaderType::Geometry => "gs_6_7",
			ShaderType::Hull => "hs_6_7",
			ShaderType::Domain => "ds_6_7",
			ShaderType::Compute => "cs_6_7",
			ShaderType::Mesh => "ms_6_7",
			ShaderType::Amplification => "as_6_7",
		}
	}
}
