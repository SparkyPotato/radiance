use std::{
	error::Error,
	ffi::OsStr,
	fs::File,
	io::{BufReader, BufWriter, Read},
	path::{Path, PathBuf},
	process::{Command, Stdio},
};

use hassle_rs::{Dxc, DxcCompiler3, DxcIncludeHandler, DxcLibrary};
use rspirv::{binary::Assemble, dr::Operand};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

pub struct ShaderModuleBuilder {
	pub(crate) vfs: VirtualFileSystem,
	dependencies: DependencyInfo,
	compiler: DxcCompiler3,
	library: DxcLibrary,
	_dxc: Dxc,
}

impl ShaderModuleBuilder {
	/// - `root` - The crate directory. Must include the `shaders/` directory in it.
	/// - `output` - The output directory. Temporary files will be written here.
	pub fn new(name: impl ToString, root: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
		let output = output.as_ref();

		let dependencies = match File::open(output.join("dependencies.json")) {
			Ok(file) => serde_json::from_reader(BufReader::new(file))?,
			Err(_) => DependencyInfo::default(),
		};

		let dxc = Dxc::new(None)?;
		let compiler = dxc.create_compiler3()?;
		let library = dxc.create_library()?;

		Ok(Self {
			vfs: VirtualFileSystem::new(name.to_string(), root.as_ref().join("shaders/"), output.into()),
			dependencies,
			compiler,
			library,
			_dxc: dxc,
		})
	}

	/// Include files from another crate's shader module.
	pub fn include(&mut self, module_path: &Path) { self.vfs.include(module_path); }

	/// Compile a specific file. The path should be relative to the `shaders/` directory.
	pub fn compile_file(&mut self, file: &Path) -> Result<(), String> {
		let ty = ShaderType::new(file)?;
		if let Some(ty) = ty {
			let args: Vec<_> = ["-spirv", "-fspv-target-env=vulkan1.3", "-HV 2021"]
				.into_iter()
				.map(|x| x.to_string())
				.chain(Some(format!("-T {}", ty.target_profile())))
				.collect();
			let args: Vec<_> = args.iter().map(|x| x.as_str()).collect();
			let mut handler = IncludeHandler::new(&self.vfs, &mut self.dependencies, file);

			let shader = std::fs::read_to_string(self.vfs.curr_path(file)).unwrap();
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

			let output = self.vfs.output.join("shaders/").join(file.with_extension("spv"));
			std::fs::create_dir_all(output.parent().unwrap()).unwrap();

			let mut spirv = rspirv::dr::load_bytes(bytecode).unwrap();
			if spirv.entry_points.len() != 1 {
				return Err(format!("Shader `{}` must have exactly one entry point", file.display()));
			}

			let mut without_hlsl = file.with_extension("");
			without_hlsl.set_extension("");
			let name = format!(
				"{}/{}",
				self.vfs.name,
				without_hlsl.to_str().expect("weird path").replace("\\", "/")
			);
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

	/// Compile all shaders in the module.
	pub fn compile_all(&mut self) -> Result<(), Vec<String>> {
		let mut errors = Vec::new();
		let mut compile_queue = Vec::new();

		for file in WalkDir::new(&self.vfs.root)
			.into_iter()
			.filter_map(|e| e.ok())
			.filter(|e| e.path().is_file())
		{
			let full_path = file.path();
			let short_path = full_path.strip_prefix(&self.vfs.root).unwrap();
			let output_path = self.vfs.output.join("shaders/").join(short_path.with_extension("spv"));

			if let Some(meta) = std::fs::metadata(output_path).ok() {
				if meta.modified().unwrap() < file.metadata().unwrap().modified().unwrap() {
					compile_queue.push(short_path.to_path_buf());
					compile_queue.extend(self.dependencies.on(&short_path).map(|x| x.to_path_buf()));
				}
			} else {
				compile_queue.push(short_path.to_path_buf());
			}
		}

		for file in compile_queue {
			eprintln!("Compiling {}", file.display());

			if let Err(e) = self.compile_file(&file) {
				errors.push(e)
			}
		}

		if errors.is_empty() {
			Ok(())
		} else {
			Err(errors)
		}
	}

	// Link all shaders into a single SPIRV module.
	pub fn link(&mut self) -> Result<(), String> {
		let files: Vec<_> = WalkDir::new(&self.vfs.output.join("shaders/"))
			.into_iter()
			.filter_map(|e| e.ok())
			.filter_map(|e| match e.path().extension().map(|x| x.to_str()).flatten() {
				Some("spv") => Some(e.path().as_os_str().to_owned()),
				_ => None,
			})
			.collect();
		let out_path = self.vfs.output.join(format!("{}.spv", self.vfs.name));
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
		if stderr.is_empty() {
			Ok(())
		} else {
			Err(stderr)
		}
	}
}

impl Drop for ShaderModuleBuilder {
	fn drop(&mut self) {
		let Ok(mut file) = File::create(self.vfs.output.join("dependencies.json")) else { return; };
		let _ = serde_json::to_writer_pretty(BufWriter::new(&mut file), &self.dependencies);
	}
}

struct IncludeHandler<'a> {
	vfs: &'a VirtualFileSystem,
	deps: &'a mut DependencyInfo,
	curr: &'a Path,
}

impl<'a> IncludeHandler<'a> {
	fn new(vfs: &'a VirtualFileSystem, deps: &'a mut DependencyInfo, curr: &'a Path) -> Self {
		Self { vfs, deps, curr }
	}

	fn load(&mut self, filename: &Path) -> Option<String> {
		match std::fs::read_to_string(&filename) {
			Ok(source) => {
				self.deps.add(self.curr, filename);
				Some(source)
			},
			Err(_) => None,
		}
	}
}

impl DxcIncludeHandler for IncludeHandler<'_> {
	fn load_source(&mut self, filename: String) -> Option<String> {
		let path = Path::new(&filename);
		let curr_dir = self.curr.parent().unwrap();
		match self.load(&curr_dir.join(path)) {
			// Check in current directory
			Some(source) => Some(source),
			None => match self.load(&self.vfs.root.join(path)) {
				// Current module root
				Some(source) => Some(source),
				None => {
					for (_, path) in self.vfs.roots.iter() {
						// Dependency roots
						match self.load(&path.join(path)) {
							Some(source) => return Some(source),
							None => continue,
						}
					}
					None
				},
			},
		}
	}
}

#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
struct DependencyInfo {
	inner: FxHashMap<PathBuf, FxHashSet<PathBuf>>,
}

impl DependencyInfo {
	pub fn add(&mut self, file: impl Into<PathBuf>, depends_on: impl Into<PathBuf>) {
		self.inner.entry(depends_on.into()).or_default().insert(file.into());
	}

	pub fn on(&self, file: impl AsRef<Path>) -> impl Iterator<Item = &Path> + '_ {
		self.inner
			.get(file.as_ref())
			.into_iter()
			.flat_map(|x| x.iter())
			.map(|x| x.as_path())
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

pub struct VirtualFileSystem {
	pub(crate) name: String,
	pub(crate) root: PathBuf,
	pub(crate) output: PathBuf,
	pub(crate) roots: FxHashMap<String, PathBuf>,
}

impl VirtualFileSystem {
	fn new(name: String, root: PathBuf, output: PathBuf) -> Self {
		Self {
			name,
			root,
			roots: FxHashMap::default(),
			output,
		}
	}

	fn include(&mut self, path: &Path) {
		let name = path.components().last().unwrap().as_os_str().to_str().unwrap();
		self.roots.insert(name.into(), path.into());
	}

	fn curr_path(&self, relative: &Path) -> PathBuf { self.root.join(relative) }
}
