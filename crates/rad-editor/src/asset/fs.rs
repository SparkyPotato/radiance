use std::{
	collections::BTreeMap,
	fs,
	io::{self, BufReader, Read, Write},
	ops::Deref,
	path::{Path, PathBuf},
	sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use rad_core::asset::{
	aref::{AssetId, UntypedAssetId},
	Asset,
	AssetRead,
	AssetSource,
	AssetWrite,
};
use rad_world::Uuid;
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::trace_span;
use walkdir::WalkDir;
use zstd::{stream::AutoFinishEncoder, Decoder, Encoder};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct AssetHeader {
	pub id: UntypedAssetId,
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

#[derive(Default)]
pub struct FsAssetSystem {
	root: RwLock<Option<PathBuf>>,
	assets: RwLock<FxHashMap<UntypedAssetId, PathBuf>>,
	by_type: RwLock<FxHashMap<Uuid, FxHashSet<UntypedAssetId>>>,
	dir: RwLock<Dir>,
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

	pub fn create<T: Asset>(&self, rel_path: &Path, id: AssetId<T>) -> Result<FsAssetWrite, io::Error> {
		let s = trace_span!("create asset", path = %rel_path.display(), id = %id);
		let _e = s.enter();

		let path = self
			.abs_path(rel_path)
			.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no system opened"))?;
		fs::create_dir_all(path.parent().unwrap())?;

		let view = FsAssetWrite::create(&path, id);
		self.add_asset(
			rel_path,
			AssetHeader {
				id: id.to_untyped(),
				ty: T::UUID,
			},
		);

		view
	}

	pub fn dir(&self) -> impl Deref<Target = Dir> + '_ { self.dir.read() }

	// pub fn assets_of_type(&self, ty: Uuid) -> FxHashSet<AssetId> {
	// 	self.by_type.read().get(&ty).cloned().unwrap_or_default()
	// }

	fn rescan(&self) {
		let s = trace_span!("rescan assets");
		let _e = s.enter();

		let r = self.root.read().clone();
		let Some(ref root) = r else {
			return;
		};
		let w = WalkDir::new(&root);

		let new = Self {
			root: RwLock::new(r),
			..Default::default()
		};
		for entry in w.into_iter().filter_map(|x| x.ok()) {
			let path = entry.path();
			let is_file = path.is_file();
			if is_file && path.extension().and_then(|x| x.to_str()) == Some("radass") {
				if let Ok(mut view) = FsAssetRead::open(entry.path()) {
					new.add_asset_abs(path, view.header());
				}
			} else if !is_file {
				new.add_dir_abs(path);
			}
		}
		*self.assets.write() = new.assets.into_inner();
		*self.by_type.write() = new.by_type.into_inner();
		*self.dir.write() = new.dir.into_inner();
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
}

impl AssetSource for FsAssetSystem {
	fn load(&self, id: UntypedAssetId, ty: Uuid) -> Result<Box<dyn AssetRead>, io::Error> {
		let s = trace_span!("load asset", id = %id, ty = %ty);
		let _e = s.enter();

		let assets = self.assets.read();
		let path = assets
			.get(&id)
			.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "asset not found"))?;
		let mut view = FsAssetRead::open(path)?;
		// TODO: cooka
		if view.header().ty != ty {
			return Err(io::Error::new(io::ErrorKind::NotFound, "asset type mismatch"));
		}
		Ok(Box::new(view))
	}
}

pub struct FsAssetRead {
	header: AssetHeader,
	read: Decoder<'static, BufReader<fs::File>>,
}
impl Read for FsAssetRead {
	fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.read.read(buf) }
}
impl AssetRead for FsAssetRead {}

impl FsAssetRead {
	fn open(path: &Path) -> Result<Self, io::Error> {
		let mut file = fs::OpenOptions::new().read(true).open(path)?;
		let mut header = AssetHeader::zeroed();
		file.read_exact(bytemuck::bytes_of_mut(&mut header))?;
		Ok(Self {
			header,
			read: Decoder::new(file)?,
		})
	}

	fn header(&mut self) -> AssetHeader { self.header }
}

pub struct FsAssetWrite {
	write: AutoFinishEncoder<'static, fs::File>,
}
impl Write for FsAssetWrite {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.write.write(buf) }

	fn flush(&mut self) -> io::Result<()> { self.write.flush() }
}
impl AssetWrite for FsAssetWrite {}

impl FsAssetWrite {
	fn create<T: Asset>(path: &Path, id: AssetId<T>) -> Result<Self, io::Error> {
		let mut file = fs::OpenOptions::new().write(true).create(true).open(path)?;
		let header = AssetHeader {
			id: id.to_untyped(),
			ty: T::UUID,
		};
		file.write_all(bytemuck::bytes_of(&header))?;
		Ok(Self {
			write: Encoder::new(file, 5)?.auto_finish(),
		})
	}
}
