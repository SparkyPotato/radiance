use std::{
	collections::BTreeMap,
	fs,
	io::{self, Read, Seek, Write},
	ops::Deref,
	path::{Path, PathBuf},
	sync::Arc,
	time::SystemTime,
};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use rad_core::asset::{AssetId, AssetSource, AssetView};
use rad_world::Uuid;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct AssetHeader {
	pub id: AssetId,
	pub ty: Uuid,
}

#[derive(Default)]
pub struct Dir {
	dirs: BTreeMap<String, Dir>,
	assets: BTreeMap<String, AssetHeader>,
}

impl Dir {
	fn add_dir(&mut self, rel_path: &Path) -> &mut Dir {
		let mut dir = self;
		for part in rel_path.iter() {
			dir = dir.dirs.entry(part.to_string_lossy().into_owned()).or_default();
		}
		dir
	}

	pub fn get_dir(&self, rel_path: &Path) -> Option<&Dir> {
		let mut dir = self;
		for part in rel_path.iter() {
			dir = dir.dirs.get(part.to_string_lossy().as_ref())?;
		}
		Some(dir)
	}

	pub fn dirs(&self) -> impl ExactSizeIterator<Item = (&String, &Dir)> + '_ { self.dirs.iter() }

	pub fn assets(&self) -> impl ExactSizeIterator<Item = (&String, &AssetHeader)> + '_ { self.assets.iter() }

	fn add_asset(&mut self, rel_path: &Path, asset: AssetHeader) {
		self.add_dir(rel_path.parent().unwrap())
			.assets
			.insert(rel_path.file_name().unwrap().to_string_lossy().into_owned(), asset);
	}
}

#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
struct ImportPaths {
	paths: FxHashMap<PathBuf, AssetId>,
}

#[derive(Default)]
pub struct FsAssetSystem {
	root: RwLock<Option<PathBuf>>,
	assets: RwLock<FxHashMap<AssetId, PathBuf>>,
	by_type: RwLock<FxHashMap<Uuid, FxHashSet<AssetId>>>,
	dir: RwLock<Dir>,
	paths: RwLock<ImportPaths>,
}

impl FsAssetSystem {
	pub fn new() -> Arc<Self> {
		let this = Arc::new(Self {
			root: RwLock::new(std::env::args().nth(1).map(PathBuf::from)),
			..Default::default()
		});
		let a = this.clone();
		// TODO: yuck
		std::thread::spawn(move || loop {
			a.rescan();
			std::thread::sleep(std::time::Duration::from_secs(5));
		});
		this
	}

	pub fn root(&self) -> impl Deref<Target = Option<PathBuf>> + '_ { self.root.read() }

	pub fn open(&self, root: PathBuf) { *self.root.write() = Some(root) }

	pub fn create(
		&self, rel_path: &Path, id: AssetId, ty: Uuid, from: Option<PathBuf>,
	) -> Result<Box<dyn AssetView>, io::Error> {
		if let Some(from) = from {
			self.paths.write().paths.insert(from, id);
		}
		let _ = self.write_imports();
		let path = self
			.abs_path(rel_path)
			.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no system opened"))?;
		fs::create_dir_all(path.parent().unwrap())?;
		let view = FsAssetView::create(&path, id, ty).map(|x| Box::new(x) as Box<_>);

		let header = AssetHeader { id, ty };
		self.add_asset(rel_path, header);

		view
	}

	pub fn import_source_of(&self, path: &Path) -> Option<(AssetId, SystemTime)> {
		let &id = self.paths.read().paths.get(path)?;
		let mtime = self.assets.read().get(&id).map(|x| {
			x.metadata()
				.ok()
				.and_then(|x| x.modified().ok())
				.unwrap_or(SystemTime::UNIX_EPOCH)
		})?;
		Some((id, mtime))
	}

	pub fn dir(&self) -> impl Deref<Target = Dir> + '_ { self.dir.read() }

	// pub fn assets_of_type(&self, ty: Uuid) -> FxHashSet<AssetId> {
	// 	self.by_type.read().get(&ty).cloned().unwrap_or_default()
	// }

	fn rescan(&self) {
		let r = self.root.read().clone();
		let Some(ref root) = r else {
			return;
		};
		let w = WalkDir::new(&root);

		let new = Self {
			root: RwLock::new(r),
			..Default::default()
		};
		let _ = new.load_imports();
		for entry in w.into_iter().filter_map(|x| x.ok()) {
			let path = entry.path();
			let is_file = path.is_file();
			if is_file && path.extension().and_then(|x| x.to_str()) == Some("radass") {
				if let Ok(mut view) = FsAssetView::open_ro(entry.path()) {
					if let Ok(header) = view.header() {
						new.add_asset_abs(path, header);
					}
				}
			} else if !is_file {
				new.add_dir_abs(path);
			}
		}
		*self.assets.write() = new.assets.into_inner();
		*self.by_type.write() = new.by_type.into_inner();
		*self.dir.write() = new.dir.into_inner();
		*self.paths.write() = new.paths.into_inner();
	}

	fn add_asset(&self, rel_path: &Path, asset: AssetHeader) {
		self.assets.write().insert(asset.id, self.abs_path(rel_path).unwrap());
		self.by_type.write().entry(asset.ty).or_default().insert(asset.id);
		self.dir.write().add_asset(rel_path, asset);
	}

	fn add_asset_abs(&self, abs_path: &Path, asset: AssetHeader) {
		self.assets.write().insert(asset.id, abs_path.to_owned());
		self.by_type.write().entry(asset.ty).or_default().insert(asset.id);
		self.dir.write().add_asset(&self.rel_path(abs_path).unwrap(), asset);
	}

	// fn add_dir(&self, rel_path: &Path) { self.dir.write().add_dir(rel_path); }

	fn add_dir_abs(&self, abs_path: &Path) { self.dir.write().add_dir(&self.rel_path(abs_path).unwrap()); }

	fn rel_path(&self, abs_path: &Path) -> Option<PathBuf> {
		self.root
			.read()
			.as_ref()
			.and_then(|x| abs_path.strip_prefix(x).ok().map(|x| x.with_extension("")))
	}

	fn abs_path(&self, rel_path: &Path) -> Option<PathBuf> {
		self.root
			.read()
			.as_ref()
			.map(|x| x.join(rel_path).with_added_extension("radass"))
	}

	fn load_imports(&self) -> Result<(), io::Error> {
		let root = self.root.read();
		let path = root
			.as_ref()
			.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no system opened"))?;
		let path = path.join(".imports.json");
		let imports = if path.exists() {
			let file = fs::File::open(&path)?;
			serde_json::from_reader(file)?
		} else {
			ImportPaths::default()
		};
		*self.paths.write() = imports;
		Ok(())
	}

	fn write_imports(&self) -> Result<(), io::Error> {
		let root = self.root.read();
		let path = root
			.as_ref()
			.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no system opened"))?;
		let path = path.join(".imports.json");
		let file = fs::File::create(&path)?;
		serde_json::to_writer_pretty(file, &*self.paths.read())?;
		Ok(())
	}
}

impl AssetSource for FsAssetSystem {
	fn load(&self, id: AssetId) -> Result<(Uuid, Box<dyn AssetView>), io::Error> {
		let mut view = FsAssetView::open(
			self.assets
				.read()
				.get(&id)
				.ok_or(io::Error::new(io::ErrorKind::NotFound, "asset not found"))?,
		)?;
		Ok((view.header()?.ty, Box::new(view)))
	}
}

struct FsAssetView {
	file: fs::File,
	name: String,
}

impl AssetView for FsAssetView {
	fn name(&self) -> &str { self.name.as_str() }

	fn clear(&mut self) -> Result<(), io::Error> { self.file.set_len(std::mem::size_of::<AssetHeader>() as _) }

	fn new_section(&mut self) -> Result<Box<dyn io::Write + '_>, io::Error> {
		Ok(Box::new(zstd::Encoder::new(&mut self.file, 16)?.auto_finish()))
	}

	fn seek_begin(&mut self) -> Result<(), io::Error> {
		self.file
			.seek(io::SeekFrom::Start(std::mem::size_of::<AssetHeader>() as _))?;
		Ok(())
	}

	fn read_section(&mut self) -> Result<Box<dyn io::Read + '_>, io::Error> {
		Ok(Box::new(zstd::Decoder::new(&mut self.file)?))
	}
}

impl FsAssetView {
	fn name(path: &Path) -> String {
		path.with_extension("")
			.file_name()
			.unwrap()
			.to_string_lossy()
			.into_owned()
	}

	fn open(path: &Path) -> Result<Self, io::Error> {
		Ok(Self {
			file: fs::OpenOptions::new().read(true).write(true).create(true).open(path)?,
			name: Self::name(path),
		})
	}

	fn open_ro(path: &Path) -> Result<Self, io::Error> {
		Ok(Self {
			file: fs::OpenOptions::new().read(true).open(path)?,
			name: Self::name(path),
		})
	}

	fn create(path: &Path, id: AssetId, ty: Uuid) -> Result<Self, io::Error> {
		let mut file = fs::OpenOptions::new().read(true).write(true).create(true).open(path)?;
		let header = AssetHeader { id, ty };
		file.write_all(bytemuck::bytes_of(&header))?;
		Ok(Self {
			file,
			name: Self::name(path),
		})
	}

	fn header(&mut self) -> Result<AssetHeader, io::Error> {
		let mut header = AssetHeader::zeroed();
		self.file.seek(io::SeekFrom::Start(0))?;
		self.file.read_exact(bytemuck::bytes_of_mut(&mut header))?;
		Ok(header)
	}
}
