//! Filesystem based assets, where each asset is a different file.

use std::{
	fs::{File, OpenOptions},
	io,
	ops::{Deref, DerefMut},
	path::{Path, PathBuf},
};

use rustc_hash::FxHashMap;
use uuid::Uuid;
use walkdir::WalkDir;

#[cfg(feature = "import")]
use crate::import::{ImportContext, ImportError, ImportProgress};
use crate::{Asset, AssetHeader, AssetSink, AssetSource, AssetSystem, AssetType};

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
		let name = path.file_name().unwrap().to_str().unwrap().to_string();
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

	fn load_header(&self) -> Result<AssetHeader, Self::Error> {
		let mut buf = [0; 30];
		self.file.read_at(0, &mut buf)?;
		let header = AssetHeader::parse(&buf, true, true)
			.ok_or(io::Error::new(io::ErrorKind::InvalidData, "Invalid asset header"))?;
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

	fn write_data(&mut self, data: &[u8]) -> Result<(), Self::Error> {
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
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AssetId(u32);

struct Directory {
	name: String,
	/// Range of child directories in `dirs`.
	children: FxHashMap<String, DirectoryId>,
	/// Range of child assets in `assets`.
	assets: FxHashMap<String, AssetId>,
}

struct AssetTree {
	/// Arena of all directories.
	dirs: Vec<Directory>,
	/// Arena of all assets.
	assets: Vec<Uuid>,
}

impl AssetTree {
	fn new(name: impl ToString) -> Self {
		Self {
			dirs: vec![Directory {
				name: name.to_string(),
				children: FxHashMap::default(),
				assets: FxHashMap::default(),
			}],
			assets: Vec::new(),
		}
	}

	fn add_asset(&mut self, rel_path: &Path, uuid: Uuid) {
		let len = self.assets.len();
		let dir = self.dir_of_mut(rel_path.parent().unwrap());
		dir.assets.insert(
			rel_path
				.with_extension("")
				.file_name()
				.unwrap()
				.to_str()
				.unwrap()
				.to_string(),
			AssetId(len as _),
		);
		self.assets.push(uuid);
	}

	fn remove_asset(&mut self, rel_path: &Path) -> Uuid {
		let dir = self.dir_of_mut(rel_path.parent().unwrap());
		let id = dir
			.assets
			.remove(rel_path.file_name().unwrap().to_str().unwrap())
			.unwrap();
		self.assets[id.0 as usize]
	}

	fn get_asset(&self, rel_path: &Path) -> Uuid {
		let dir = self.dir_of(rel_path.parent().unwrap());
		let &id = dir.assets.get(rel_path.file_name().unwrap().to_str().unwrap()).unwrap();
		self.assets[id.0 as usize]
	}

	fn dir_of(&self, rel_path: &Path) -> &Directory {
		let mut curr = 0;
		let mut components = rel_path.components();
		if rel_path.is_file() {
			components.next_back();
		}
		for comp in components {
			let name = comp.as_os_str().to_str().unwrap();
			let &idx = self.dirs[curr].children.get(name).unwrap();
			curr = idx.0 as usize;
		}
		&self.dirs[curr]
	}

	fn dir_of_mut(&mut self, rel_path: &Path) -> &mut Directory {
		let mut curr = 0;
		let mut components = rel_path.components();
		if rel_path.is_file() {
			components.next_back();
		}
		for comp in components {
			let name = comp.as_os_str().to_str().unwrap();
			curr = match self.dirs[curr].children.get(name) {
				Some(&idx) => idx.0 as usize,
				None => {
					let idx = self.dirs.len();
					self.dirs.push(Directory {
						name: name.to_string(),
						children: FxHashMap::default(),
						assets: FxHashMap::default(),
					});
					self.dirs[curr].children.insert(name.to_string(), DirectoryId(idx as _));
					idx
				},
			};
		}
		&mut self.dirs[curr]
	}
}

pub struct DirView<'a> {
	tree: &'a AssetTree,
	dir: DirectoryId,
}

impl DirView<'_> {
	pub fn name(&self) -> &str { &self.tree.dirs[self.dir.0 as usize].name }

	pub fn dirs(&self) -> impl Iterator<Item = DirView<'_>> {
		let dir = &self.tree.dirs[self.dir.0 as usize];
		dir.children.iter().map(move |(_, &id)| DirView {
			tree: self.tree,
			dir: id,
		})
	}

	pub fn assets(&self) -> impl Iterator<Item = (&str, Uuid)> + '_ {
		let dir = &self.tree.dirs[self.dir.0 as usize];
		dir.assets
			.iter()
			.map(move |(name, &id)| (name.as_str(), self.tree.assets[id.0 as usize]))
	}
}

/// A filesystem-based asset system.
pub struct FsSystem {
	root: PathBuf,
	tree: AssetTree,
	system: AssetSystem<FsAsset>,
	#[cfg(feature = "import")]
	file_conflict_map: FxHashMap<PathBuf, u32>,
}

impl FsSystem {
	// Create a new filesystem-based asset system, recursively scanning the given root directory for assets.
	pub fn new(root: impl Into<PathBuf>) -> Self {
		let root = root.into();
		let mut system = AssetSystem::new();
		let mut tree = AssetTree::new(root.components().last().unwrap().as_os_str().to_str().unwrap());

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
			file_conflict_map: FxHashMap::default(),
		}
	}

	/// Add an asset copied into the root directory or a subdirectory.
	pub fn add(&mut self, path: impl AsRef<Path>) -> Result<AssetHeader, io::Error> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let header = self.system.add(FsAsset::load(path)?)?;
		self.tree.add_asset(rel_path, header.uuid);
		Ok(header)
	}

	/// Remove an asset.
	pub fn remove(&mut self, path: impl AsRef<Path>) -> AssetHeader {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.remove_asset(rel_path);
		let header = self.system.remove(uuid).0;
		let _ = std::fs::remove_file(path);
		header
	}

	/// Load an asset by path.
	pub fn load(&mut self, path: impl AsRef<Path>) -> Result<Asset, io::Error> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.get_asset(rel_path);
		self.system.load(uuid)
	}

	/// Write an asset to disk.
	pub fn write(&mut self, path: impl AsRef<Path>, asset: Asset) -> Result<(), io::Error> {
		let path = path.as_ref();
		let rel_path = path.strip_prefix(&self.root).expect("Asset path must be inside root");
		let uuid = self.tree.get_asset(rel_path);
		self.system.write(uuid, asset)
	}

	/// Import an asset from a file, at `out_path`.
	#[cfg(feature = "import")]
	pub fn import(
		&mut self, path: impl AsRef<Path>, out_path: impl AsRef<Path>,
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
					if let Some(p) = self.file_conflict_map.get_mut(&path) {
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
			let dir = &self.tree.dirs[curr.0 as usize];
			if let Some(id) = dir.children.get(name) {
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
