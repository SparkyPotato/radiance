use std::{
	borrow::Borrow,
	error::Error,
	fmt::Display,
	ops::Deref,
	path::{Path, PathBuf},
};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[repr(transparent)]
#[derive(Eq, PartialEq, Hash)]
pub struct VirtualPath {
	inner: Path,
}

impl VirtualPath {
	pub fn new<P: AsRef<Path> + ?Sized>(path: &P) -> &Self {
		unsafe { &*(path.as_ref() as *const Path as *const VirtualPath) }
	}

	pub fn get_module(&self) -> &str { self.inner.components().next().unwrap().as_os_str().to_str().unwrap() }

	pub fn display(&self) -> impl Display + '_ { self.inner.to_str().expect("weird path").replace('\\', "/") }
}

#[repr(transparent)]
#[derive(Eq, PartialEq, Hash, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VirtualPathBuf {
	inner: PathBuf,
}

impl VirtualPathBuf {
	pub fn new(path: impl Into<PathBuf>) -> Self { Self { inner: path.into() } }
}

impl Borrow<VirtualPath> for VirtualPathBuf {
	fn borrow(&self) -> &VirtualPath { VirtualPath::new(&self.inner) }
}

impl From<&'_ VirtualPath> for VirtualPathBuf {
	fn from(path: &'_ VirtualPath) -> Self {
		Self {
			inner: path.inner.to_owned(),
		}
	}
}

impl Deref for VirtualPathBuf {
	type Target = VirtualPath;

	fn deref(&self) -> &Self::Target { VirtualPath::new(&self.inner) }
}

impl AsRef<VirtualPath> for &'_ VirtualPathBuf {
	fn as_ref(&self) -> &VirtualPath { VirtualPath::new(&self.inner) }
}

#[derive(Default)]
pub struct VirtualFileSystem {
	source_roots: FxHashMap<String, PathBuf>,
	source_reverse: FxHashMap<PathBuf, String>,
	output_roots: FxHashMap<String, Option<PathBuf>>,
	output_reverse: FxHashMap<PathBuf, String>,
}

impl VirtualFileSystem {
	pub fn new() -> Self { Self::default() }

	/// Add a root directory to the virtual file system.
	///
	/// `source_path` must be a directory containing a `Cargo.toml` file and a `shaders` directory.
	pub fn add_root<T: Into<PathBuf>>(&mut self, source_path: T, output_path: Option<T>) -> Result<(), Box<dyn Error>> {
		let mut source_path = source_path.into();
		let mut output_path = output_path.map(Into::into);

		let name = get_cargo_package_name(&source_path)?;
		source_path.push("shaders");
		self.source_roots.insert(name.clone(), source_path.clone());
		self.source_reverse.insert(source_path, name.clone());

		if let Some(output_path) = &mut output_path {
			output_path.push("shaders");
			self.output_roots.insert(name.clone(), Some(output_path.clone()));
			self.output_reverse.insert(output_path.clone(), name);
		} else {
			self.output_roots.insert(name, None);
		}

		Ok(())
	}

	/// Get the physical source root of a module.
	pub fn get_root(&self, name: &str) -> Option<&Path> { self.source_roots.get(name).map(|p| p.as_ref()) }

	/// Get the physical output root of a module.
	pub fn get_output(&self, name: &str) -> Option<&Path> {
		self.output_roots.get(name).and_then(|p| p.as_ref().map(|x| x.as_ref()))
	}

	/// Convert a virtual path to a physical source path.
	pub fn resolve_source(&self, path: &VirtualPath) -> Option<PathBuf> {
		let mut components = path.inner.components();
		let name = components.next()?;
		let name = name.as_os_str().to_str()?;
		let root = self.source_roots.get(name)?;
		let path = components.as_path();

		Some(root.join(path).with_extension("hlsl"))
	}

	/// Convert a physical source path to a virtual path.
	pub fn unresolve_source(&self, path: &Path) -> Option<VirtualPathBuf> {
		let mut test = path;
		loop {
			if let Some(name) = self.source_reverse.get(test) {
				let mut path = Path::new(name).join(path.strip_prefix(test).ok()?);
				path.set_extension("");
				return Some(VirtualPathBuf::new(path));
			}

			test = test.parent()?;
		}
	}

	/// Convert a virtual path to a physical output path.
	pub fn resolve_output(&self, path: &VirtualPath) -> Option<PathBuf> {
		let mut components = path.inner.components();
		let name = components.next()?;
		let name = name.as_os_str().to_str()?;
		let root = self.output_roots.get(name)?.as_ref()?;
		let path = components.as_path();

		Some(root.join(path).with_extension("spv"))
	}

	/// Convert a physical output path to a virtual path.
	pub fn unresolve_output(&self, path: &Path) -> Option<VirtualPathBuf> {
		let mut test = path;
		loop {
			if let Some(name) = self.output_reverse.get(test) {
				let path = Path::new(name).join(path.strip_prefix(test).ok()?).with_extension("");
				return Some(VirtualPathBuf::new(path));
			}

			test = test.parent()?;
		}
	}

	pub fn compilable_modules(&self) -> impl Iterator<Item = (&str, &Path, &Path)> + '_ {
		self.source_roots.iter().filter_map(|(name, source)| {
			self.output_roots[name]
				.as_ref()
				.map(|out| (name.as_ref(), source.as_path(), out.as_path()))
		})
	}
}

fn get_cargo_package_name(root: &Path) -> Result<String, Box<dyn Error>> {
	let manifest = root.join("Cargo.toml");
	let manifest = cargo_toml::Manifest::from_path(manifest)?;
	let name = manifest.package.ok_or("Cargo.toml has no package")?.name;
	Ok(name)
}
