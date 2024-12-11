use std::{
	collections::BTreeMap,
	fs,
	io::{self, Read, Seek, Write},
	ops::Deref,
	path::{Path, PathBuf},
	sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use rad_core::asset::{AssetId, AssetSource, AssetView};
use rad_world::Uuid;
use rustc_hash::{FxHashMap, FxHashSet};
use walkdir::WalkDir;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct AssetHeader {
	pub id: AssetId,
	pub ty: Uuid,
}

pub struct Dir {
	dirs: BTreeMap<String, Dir>,
	assets: BTreeMap<String, AssetHeader>,
}

impl Dir {
	fn new() -> Self {
		Self {
			dirs: BTreeMap::new(),
			assets: BTreeMap::new(),
		}
	}

	fn add_dir(&mut self, rel_path: &Path) -> &mut Dir {
		let mut dir = self;
		for part in rel_path.iter() {
			dir = dir
				.dirs
				.entry(part.to_string_lossy().into_owned())
				.or_insert_with(Dir::new);
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

pub struct FsAssetSystem {
	root: RwLock<Option<PathBuf>>,
	assets: RwLock<FxHashMap<AssetId, PathBuf>>,
	by_type: RwLock<FxHashMap<Uuid, FxHashSet<AssetId>>>,
	dir: RwLock<Dir>,
}

impl FsAssetSystem {
	pub fn new() -> Arc<Self> {
		let this = Arc::new(Self {
			root: RwLock::new(std::env::args().nth(1).map(PathBuf::from)),
			assets: RwLock::new(FxHashMap::default()),
			by_type: RwLock::new(FxHashMap::default()),
			dir: RwLock::new(Dir::new()),
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

	pub fn create(&self, path: &Path, id: AssetId, ty: Uuid) -> Result<Box<dyn AssetView>, io::Error> {
		let path = self
			.root
			.read()
			.as_ref()
			.map(|x| x.join(path.with_extension("radass")))
			.ok_or(io::Error::new(io::ErrorKind::NotFound, "no system opened"))?;
		fs::create_dir_all(path.parent().unwrap())?;
		FsAssetView::create(&path, id, ty).map(|x| Box::new(x) as Box<_>)
	}

	pub fn dir(&self) -> impl Deref<Target = Dir> + '_ { self.dir.read() }

	// pub fn assets_of_type(&self, ty: Uuid) -> FxHashSet<AssetId> {
	// 	self.by_type.read().get(&ty).cloned().unwrap_or_default()
	// }

	pub(super) fn rescan(&self) {
		let r = self.root.read().clone();
		let Some(root) = r else {
			return;
		};
		let w = WalkDir::new(&root);

		let mut assets = FxHashMap::default();
		let mut by_type: FxHashMap<Uuid, FxHashSet<AssetId>> = FxHashMap::default();
		let mut dir = Dir::new();
		for entry in w
			.into_iter()
			.filter_map(|x| x.ok())
			.filter(|x| x.path().extension().map(|x| x == "radass").unwrap_or(false))
		{
			let path = entry.path();
			let is_file = path.is_file();
			if is_file && path.extension().and_then(|x| x.to_str()) == Some("radass") {
				if let Ok(mut view) = FsAssetView::open_ro(entry.path()) {
					if let Ok(header) = view.header() {
						assets.insert(header.id, entry.path().to_path_buf());
						by_type.entry(header.ty).or_default().insert(header.id);
						dir.add_asset(&path.strip_prefix(&root).unwrap().with_extension(""), header);
					}
				}
			} else if !is_file {
				dir.add_dir(&path.strip_prefix(&root).unwrap().with_extension(""));
			}
		}
		*self.assets.write() = assets;
		*self.by_type.write() = by_type;
		*self.dir.write() = dir;
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
}

impl AssetView for FsAssetView {
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
	fn open(path: &Path) -> Result<Self, io::Error> {
		Ok(Self {
			file: fs::OpenOptions::new().read(true).write(true).create(true).open(path)?,
		})
	}

	fn open_ro(path: &Path) -> Result<Self, io::Error> {
		Ok(Self {
			file: fs::OpenOptions::new().read(true).open(path)?,
		})
	}

	fn create(path: &Path, id: AssetId, ty: Uuid) -> Result<Self, io::Error> {
		let mut file = fs::OpenOptions::new().read(true).write(true).create(true).open(path)?;
		let header = AssetHeader { id, ty };
		file.write_all(bytemuck::bytes_of(&header))?;
		Ok(Self { file })
	}

	fn header(&mut self) -> Result<AssetHeader, io::Error> {
		let mut header = AssetHeader::zeroed();
		self.file.seek(io::SeekFrom::Start(0))?;
		self.file.read_exact(bytemuck::bytes_of_mut(&mut header))?;
		Ok(header)
	}
}
