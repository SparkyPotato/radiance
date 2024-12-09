use std::{
	fs,
	io::{self, Read, Seek, Write},
	path::{Path, PathBuf},
	sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use rad_core::asset::{AssetId, AssetSource, AssetView};
use rad_world::Uuid;
use rustc_hash::FxHashMap;
use walkdir::WalkDir;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct AssetHeader {
	id: AssetId,
	ty: Uuid,
}

pub struct FsAssetSystem {
	root: RwLock<Option<PathBuf>>,
	assets: RwLock<FxHashMap<AssetId, PathBuf>>,
}

impl FsAssetSystem {
	pub fn new() -> Arc<Self> {
		let this = Arc::new(Self {
			root: RwLock::new(std::env::args().nth(1).map(PathBuf::from)),
			assets: RwLock::new(FxHashMap::default()),
		});
		let a = this.clone();
		// TODO: yuck
		std::thread::spawn(move || loop {
			a.rescan();
			std::thread::sleep(std::time::Duration::from_secs(5));
		});
		this
	}

	pub fn is_open(&self) -> bool { self.root.read().is_some() }

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

	fn rescan(&self) {
		let r = self.root.read();
		let Some(root) = r.as_ref() else {
			return;
		};

		let mut assets = FxHashMap::default();
		let w = WalkDir::new(root);
		drop(r);
		for entry in w
			.into_iter()
			.filter_map(|x| x.ok())
			.filter(|x| x.path().extension().map(|x| x == "radass").unwrap_or(false))
		{
			if let Ok(mut view) = FsAssetView::open_ro(entry.path()) {
				if let Ok(header) = view.header() {
					assets.insert(header.id, entry.path().to_path_buf());
				}
			}
		}
		*self.assets.write() = assets;
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
