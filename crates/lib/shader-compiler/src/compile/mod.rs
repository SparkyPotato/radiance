use std::{
	borrow::Borrow,
	collections::HashSet,
	error::Error,
	fs::File,
	io::{self, BufReader},
	path::Path,
	process::{Child, Command, Stdio},
};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::compile::vfs::{VirtualFileSystem, VirtualPath, VirtualPathBuf};

pub mod vfs;

pub struct ShaderBuilder {
	pub(crate) vfs: VirtualFileSystem,
	debug: bool,
	dependencies: DependencyInfo,
}

impl ShaderBuilder {
	pub fn new(debug: bool) -> Result<Self, Box<dyn Error>> {
		Ok(Self {
			vfs: VirtualFileSystem::new(),
			debug: true,
			dependencies: DependencyInfo::default(),
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
	pub fn compile_file_physical(&mut self, file: &Path) -> Result<Child, io::Error> {
		let virtual_path = self.vfs.unresolve_source(file).unwrap();

		let output = self.vfs.resolve_output(&virtual_path).unwrap();
		std::fs::create_dir_all(output.parent().unwrap())?;
		Command::new("slangc")
			.arg(file)
			.args([
				"-target",
				"spirv",
				"-profile",
				"sm_6_7",
				"-fvk-use-scalar-layout",
				"-matrix-layout-column-major",
				"-O2",
				"-g0",
			])
			.args(self.vfs.includes().flat_map(|p| [Path::new("-I"), p]))
			.args([Path::new("-o"), output.as_path()])
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
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

		let mut children = Vec::new();
		for file in compile_queue {
			match self.compile_file_physical(&file) {
				Ok(x) => children.push((x, file)),
				Err(e) => errors.push(format!("{}:\n{}", file.display(), e)),
			}
		}

		for (child, file) in children {
			match child.wait_with_output() {
				Ok(x) => {
					if !x.stdout.is_empty() || !x.stderr.is_empty() {
						errors.push(String::from_utf8(x.stdout).unwrap());
						errors.push(String::from_utf8(x.stderr).unwrap());
					}
				},
				Err(e) => errors.push(format!("{}:\n{}", file.display(), e)),
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
