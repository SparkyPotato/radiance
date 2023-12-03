//! Filesystem based assets, where each asset is a different file.

use std::{
	fs::{File, OpenOptions},
	hash::BuildHasherDefault,
	io,
	ops::{Deref, DerefMut},
	path::{Path, PathBuf},
};

use dashmap::DashMap;
use parking_lot::{lock_api::MappedRwLockReadGuard, RawRwLock, RwLock, RwLockReadGuard};
use uuid::Uuid;
use walkdir::WalkDir;

#[cfg(feature = "import")]
use crate::import::{ImportContext, ImportError, ImportProgress};
use crate::{Asset, AssetError, AssetHeader, AssetSink, AssetSource, AssetSystem, AssetType, HeaderParseError};

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;

struct PlatformFile {
	file: File,
}

impl PlatformFile {
	fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
		#[cfg(target_os = "unix")]
		{
			use std::os::unix::fs::FileExt;
			self.file.read_at(buf, offset)
		}
		#[cfg(target_os = "windows")]
		{
			use std::os::windows::fs::FileExt;
			self.file.seek_read(buf, offset)
		}
	}

	fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<usize> {
		#[cfg(target_os = "unix")]
		{
			use std::os::unix::fs::FileExt;
			self.file.write_at(buf, offset)
		}
		#[cfg(target_os = "windows")]
		{
			use std::os::windows::fs::FileExt;
			self.file.seek_write(buf, offset)
		}
	}
}

/// An asset source backed by a file.
pub struct FsAsset {
	name: String,
	file: PlatformFile,
}

impl FsAsset {
	pub fn load(path: &Path) -> Result<Self, io::Error> {
		let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
		let name = path
			.file_name()
			.unwrap()
			.to_str()
			.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-string file name"))?
			.to_string();
		Ok(Self {
			name,
			file: PlatformFile { file },
		})
	}

	pub fn create(path: &Path, header: AssetHeader) -> Result<Self, io::Error> {
		let this = Self::load(path)?;
		this.file.write_at(0, &header.to_bytes(true, true))?;
		Ok(this)
	}
}

impl AssetSource for FsAsset {
	type Error = io::Error;

	fn human_name(&self) -> Option<&str> { Some(&self.name) }

	fn load_header(&self) -> Result<Result<AssetHeader, HeaderParseError>, Self::Error> {
		let mut buf = [0; 30];
		self.file.read_at(0, &mut buf)?;
		let header = AssetHeader::parse(&buf, true, true);
		Ok(header)
	}

	fn load_data(&self) -> Result<Vec<u8>, Self::Error> {
		let mut buf = vec![0; self.file.file.metadata()?.len() as usize - 30];
		self.file.read_at(30, &mut buf)?;
		Ok(buf)
	}
}

impl AssetSink for FsAsset {
	type Error = io::Error;

	fn write_data(&self, data: &[u8]) -> Result<(), Self::Error> {
		self.file.file.set_len(30 + data.len() as u64)?;
		self.file.write_at(30, data)?;
		Ok(())
	}
}

#[cfg(feature = "import")]
pub struct FsImporter<'a, F, P, C>
where
	F: FnMut(&Path, Uuid),
	P: Fn(ImportProgress, ImportProgress),
	C: FnMut(PathBuf) -> PathBuf,
{
	out_path: &'a Path,
	on_import: F,
	on_conflict: C,
	progress: P,
}

#[cfg(feature = "import")]
impl<F, P, C> ImportContext for FsImporter<'_, F, P, C>
where
	F: FnMut(&Path, Uuid) + Send + Sync,
	P: Fn(ImportProgress, ImportProgress) + Send + Sync,
	C: FnMut(PathBuf) -> PathBuf + Send + Sync,
{
	type Error = io::Error;
	type Sink = FsAsset;

	fn asset(&mut self, name: &str, header: AssetHeader) -> Result<Self::Sink, Self::Error> {
		let inter = match header.ty {
			AssetType::Mesh => "meshes",
			AssetType::Model => "models",
			AssetType::Material => "materials",
			AssetType::Scene => "scenes",
		};

		let mut path = self.out_path.join(inter);
		std::fs::create_dir_all(&path)?;
		path.push(name);
		path.set_extension("radass");

		if path.exists() {
			path = (self.on_conflict)(path);
		}
		debug_assert!(
			!path.exists(),
			"Conflict resolution function returned a path that already exists"
		);

		(self.on_import)(&path, header.uuid);
		FsAsset::create(&path, header)
	}

	fn progress(&self, progress: ImportProgress, total: ImportProgress) { (self.progress)(progress, total) }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DirectoryId(u32);

struct Directory {
	name: String,
	children: FxDashMap<String, DirectoryId>,
	assets: FxDashMap<String, Uuid>,
}

struct AssetTree {
	/// Arena of all directories.
	dirs: RwLock<Vec<Directory>>,
}

impl AssetTree {
	fn new(name: impl ToString) -> Self {
		Self {
			dirs: RwLock::new(vec![Directory {
				name: name.to_string(),
				children: FxDashMap::default(),
				assets: FxDashMap::default(),
			}]),
		}
	}

	fn add_asset(&self, rel_path: &Path, uuid: Uuid) {
		let dir = self.add_dir(rel_path.parent().unwrap());
		dir.assets.insert(
			rel_path
				.with_extension("")
				.file_name()
				.unwrap()
				.to_str()
				.unwrap()
				.to_string(),
			uuid,
		);
	}

	fn remove_asset(&self, rel_path: &Path) -> Uuid {
		let dir = self.dir_of(rel_path.parent().unwrap());
		let (_, id) = dir
			.assets
			.remove(rel_path.file_name().unwrap().to_str().unwrap())
			.unwrap();
		id
	}

	fn get_asset(&self, rel_path: &Path) -> Uuid {
		let dir = self.dir_of(rel_path.parent().unwrap());
		let x = *dir.assets.get(rel_path.file_name().unwrap().to_str().unwrap()).unwrap();
		x
	}

	fn dir_of(&self, rel_path: &Path) -> MappedRwLockReadGuard<RawRwLock, Directory> {
		let mut curr = 0;
		let mut components = rel_path.components();
		if rel_path.is_file() {
			components.next_back();
		}
		for comp in components {
			let name = comp.as_os_str().to_str().unwrap();
			let d = self.dirs.read();
			let r = d[curr].children.get(name).unwrap();
			curr = r.0 as usize;
		}
		RwLockReadGuard::map(self.dirs.read(), |x| &x[curr])
	}

	fn add_dir(&self, rel_path: &Path) -> MappedRwLockReadGuard<RawRwLock, Directory> {
		let mut curr = 0;
		let mut components = rel_path.components();
		if rel_path.is_file() {
			components.next_back();
		}
		for comp in components {
			let name = comp.as_os_str().to_str().unwrap();
			let dirs = self.dirs.read();
			let x = dirs[curr].children.get(name);
			curr = match x {
				Some(idx) => idx.0 as usize,
				None => {
					let idx = dirs.len();
					dirs[curr].children.insert(name.to_string(), DirectoryId(idx as _));
					drop(x);
					drop(dirs);
					self.dirs.write().push(Directory {
						name: name.to_string(),
						children: FxDashMap::default(),
						assets: FxDashMap::default(),
					});
					idx
				},
			};
		}
		RwLockReadGuard::map(self.dirs.read(), |x| &x[curr])
	}
}

pub struct DirView<'a> {
	tree: &'a AssetTree,
	dir: DirectoryId,
}

impl DirView<'_> {
	pub fn name(&self) -> MappedRwLockReadGuard<RawRwLock, str> {
		RwLockReadGuard::map(self.tree.dirs.read(), |x| x[self.dir.0 as usize].name.as_str())
	}

	pub fn for_each_dir(&self, mut f: impl FnMut(DirView<'_>)) {
		for r in self.tree.dirs.read()[self.dir.0 as usize].children.iter() {
			f(DirView {
				tree: self.tree,
				dir: *r,
			});
		}
	}

	pub fn for_each_asset(&self, mut f: impl FnMut(&str, Uuid)) {
		for r in self.tree.dirs.read()[self.dir.0 as usize].assets.iter() {
			f(r.key(), *r);
		}
	}

	pub fn elems(&self) -> usize {
		let dirs = self.tree.dirs.read();
		let dir = &dirs[self.dir.0 as usize];
		dir.children.len() + dir.assets.len()
	}
}

/// A filesystem-based asset system.
pub struct FsSystem {
	root: PathBuf,
	tree: AssetTree,
	system: AssetSystem<FsAsset>,
	#[cfg(feature = "import")]
	file_conflict_map: FxDashMap<PathBuf, u32>,
}

impl FsSystem {
	// Create a new filesystem-based asset system, recursively scanning the given root directory for assets.
	pub fn new(root: impl Into<PathBuf>) -> Self {
		let root = root.into();
		let system = AssetSystem::new();
		let tree = AssetTree::new(root.components().last().unwrap().as_os_str().to_str().unwrap());

		for file in WalkDir::new(&root)
			.into_iter()
			.flat_map(|x| x.ok())
			.filter(|x| x.path().is_file())
		{
			let path = file.path();
			let header = match path.extension().and_then(|x| x.to_str()) {
				Some("radass") => {
					if let Ok(asset) = FsAsset::load(path) {
						system.add(asset).ok()
					} else {
						panic!("bad asset");
					}
				},
				_ => None,
			};

			if let Some(header) = header {
				let rel_path = path.strip_prefix(&root).unwrap();
				tree.add_asset(rel_path, header.uuid);
			}
		}

		Self {
			root,
			tree,
			system,
			#[cfg(feature = "import")]
			file_conflict_map: FxDashMap::default(),
		}
	}

	/// Add an asset copied into the root directory or a subdirectory.
	pub fn add(&self, path: impl AsRef<Path>) -> Result<AssetHeader, AssetError<FsAsset>> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let header = self
			.system
			.add(FsAsset::load(path).map_err(|x| AssetError::Source(x))?)?;
		self.tree.add_asset(rel_path, header.uuid);
		Ok(header)
	}

	/// Remove an asset.
	pub fn remove(&self, path: impl AsRef<Path>) -> AssetHeader {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.remove_asset(rel_path);
		let header = self.system.remove(uuid).0;
		let _ = std::fs::remove_file(path);
		header
	}

	/// Load an asset by path.
	pub fn load(&self, path: impl AsRef<Path>) -> Result<Asset, AssetError<FsAsset>> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.get_asset(rel_path);
		self.system.load(uuid)
	}

	/// Write an asset to disk.
	pub fn write(&self, path: impl AsRef<Path>, asset: Asset) -> Result<(), io::Error> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.get_asset(rel_path);
		self.system.write(uuid, asset)
	}

	/// Import an asset from a file, at `out_path`.
	#[cfg(feature = "import")]
	pub fn import(
		&self, path: impl AsRef<Path>, out_path: impl AsRef<Path>,
		progress: impl Fn(ImportProgress, ImportProgress) + Send + Sync,
	) -> Result<(), ImportError<io::Error, io::Error>> {
		let path = path.as_ref();
		let mut out_path = out_path.as_ref().to_path_buf();
		out_path.push(path.file_name().unwrap());
		out_path.set_extension("");

		self.system.import(
			FsImporter {
				out_path: &out_path,
				on_import: |path, uuid| self.tree.add_asset(path.strip_prefix(&self.root).unwrap(), uuid),
				on_conflict: |mut path| {
					if let Some(mut p) = self.file_conflict_map.get_mut(&path) {
						let stem = path.file_stem().unwrap().to_str().unwrap();
						path.set_file_name(format!("{}_{}.radass", stem, *p));
						*p += 1;
						path
					} else {
						let mut p = 1;
						let stem = path.file_stem().unwrap().to_str().unwrap();
						let len = stem.bytes().rev().take_while(|x| x.is_ascii_digit()).count();
						let stem = stem[..stem.len() - len].to_string();
						while path.exists() {
							path.set_file_name(format!("{}_{}.radass", stem, p));
							p += 1;
						}
						self.file_conflict_map.insert(path.clone(), p);
						path
					}
				},
				progress,
			},
			path,
		)?;
		Ok(())
	}

	pub fn root(&self) -> &Path { &self.root }

	pub fn id_of_dir(&self, path: impl AsRef<Path>) -> Option<DirectoryId> {
		let mut curr = DirectoryId(0);
		for component in path.as_ref().strip_prefix(&self.root).ok()?.components() {
			let name = component.as_os_str().to_str().unwrap();
			if let Some(id) = self.tree.dirs.read()[curr.0 as usize].children.get(name) {
				curr = *id;
			} else {
				return None;
			}
		}
		Some(curr)
	}

	pub fn dir_view(&self, path: impl AsRef<Path>) -> Option<DirView<'_>> {
		self.id_of_dir(path).map(|dir| DirView { tree: &self.tree, dir })
	}
}

impl Deref for FsSystem {
	type Target = AssetSystem<FsAsset>;

	fn deref(&self) -> &Self::Target { &self.system }
}

impl DerefMut for FsSystem {
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.system }
}
