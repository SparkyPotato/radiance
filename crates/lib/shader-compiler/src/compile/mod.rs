use std::{borrow::Borrow, collections::HashSet, error::Error, fs::File, io::BufReader, path::Path};

use hassle_rs::{Dxc, DxcCompiler, DxcIncludeHandler, DxcLibrary};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::compile::vfs::{VirtualFileSystem, VirtualPath, VirtualPathBuf};

pub mod vfs;

pub struct ShaderBuilder {
	pub(crate) vfs: VirtualFileSystem,
	debug: bool,
	dependencies: DependencyInfo,
	compiler: DxcCompiler,
	library: DxcLibrary,
	_dxc: Dxc,
}

impl ShaderBuilder {
	pub fn new(debug: bool) -> Result<Self, Box<dyn Error>> {
		let dxc = Dxc::new(None)?;
		let compiler = dxc.create_compiler()?;
		let library = dxc.create_library()?;

		Ok(Self {
			vfs: VirtualFileSystem::new(),
			debug: true,
			dependencies: DependencyInfo::default(),
			compiler,
			library,
			_dxc: dxc,
		})
	}

	/// Load dependencies from a file.
	pub fn deps(&mut self, file: &Path) -> Result<(), Box<dyn Error>> {
		let mut file = BufReader::new(File::open(file)?);
		let deps = serde_json::from_reader(&mut file)?;
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
			.chain(
				if self.debug { Some(["-Zi", "-Od"]) } else { None }
					.into_iter()
					.flatten(),
			)
			.map(|x| x.to_string())
			.chain(Some(format!("-T {}", ty.target_profile())))
			.collect();
			let args: Vec<_> = args.iter().map(|x| x.as_str()).collect();
			let mut handler = IncludeHandler::new(&self.vfs, &mut self.dependencies, &virtual_path);

			let shader = std::fs::read_to_string(file).map_err(|x| x.to_string())?;
			let blob = self
				.library
				.create_blob_with_encoding_from_str(&shader)
				.map_err(|x| x.to_string())?;
			let bytecode: Vec<u8> = match self.compiler.compile(
				&blob,
				&file.file_name().unwrap().to_string_lossy(),
				"main",
				ty.target_profile(),
				&args,
				Some(&mut handler),
				&[],
			) {
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
			std::fs::write(output, bytecode).unwrap();
		}

		Ok(())
	}

	/// Compile all shaders in all modules.
	pub fn compile_all(&mut self) -> Result<(), Vec<String>> {
		let mut errors = Vec::new();
		let mut compile_queue = HashSet::new();

		for (_, module_root, _) in self.vfs.compilable_modules() {
			for file in WalkDir::new(module_root)
				.into_iter()
				.filter_map(|e| e.ok())
				.filter(|e| e.path().is_file())
			{
				let path = file.path();
				let virtual_path = self.vfs.unresolve_source(path).unwrap();
				let output_path = self.vfs.resolve_output(&virtual_path).unwrap();

				let modified = output_path
					.metadata()
					.ok()
					.and_then(|o| file.metadata().ok().map(|f| (f, o)))
					.map(|(f, o)| f.modified().unwrap() > o.modified().unwrap())
					.unwrap_or(true);
				if modified {
					compile_queue.insert(path.to_path_buf());
					compile_queue.extend(self.dependencies.on(&virtual_path).filter_map(|x| {
						let path = self.vfs.resolve_source(x).unwrap();
						path.exists().then(|| path)
					}));
				}
			}
		}

		for file in compile_queue {
			if let Err(e) = self.compile_file_physical(&file) {
				errors.push(format!("{}:\n{}", file.display(), e))
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

		// Check in current directory
		match self.load(&curr_dir.join(path)) {
			Some(source) => Some(source),
			// Global include directories
			None => {
				let path = path.with_extension("");
				self.load(&self.vfs.resolve_source(VirtualPath::new(&path))?)
			},
		}
	}
}

#[derive(Default, Debug, Serialize, Deserialize)]
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
	RayTracing,
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
			"r" => Some(ShaderType::RayTracing),
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
			ShaderType::RayTracing => "lib_6_7",
		}
	}
}

