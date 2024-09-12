use std::{
	any::Any,
	error::Error,
	fmt::{Debug, Display},
	fs::{File, OpenOptions},
	hash::BuildHasherDefault,
	io::Read,
	path::{Component, Path, PathBuf},
	sync::Arc,
};

use bytemuck::{checked::from_bytes, Pod, Zeroable};
use crossbeam_channel::{Receiver, Sender};
use dashmap::{mapref::entry::Entry, DashMap};
use radiance_graph::{device::Device, graph::Frame};
use rustc_hash::{FxHashMap, FxHasher};
pub use uuid::Uuid;
use walkdir::WalkDir;

use crate::{
	io::{Reader, Writer},
	rref::{DelRes, RRef, RWeak},
};

pub mod gltf;
pub mod io;
pub mod mesh;
pub mod rref;
pub mod scene;

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct AssetHeader {
	pub ty: Uuid,
	pub asset: Uuid,
}

impl AssetHeader {
	fn from_path(path: &Path) -> Result<Self, std::io::Error> { Self::from_file(&mut File::open(path)?) }

	fn from_file(file: &mut File) -> Result<Self, std::io::Error> {
		let mut out = [0u8; std::mem::size_of::<Self>()];
		file.read_exact(&mut out)?;

		Ok(*from_bytes(&out))
	}
}

pub type LResult<T> = Result<RRef<T>, LoadError>;

pub struct InitContext<'a, T> {
	pub name: &'a str,
	pub device: &'a Device,
	pub uuid: Uuid,
	pub data: Reader,
	pub sys: &'a AssetSystem,
	pub runtime: Arc<T>,
}

impl<C> InitContext<'_, C> {
	fn make<T: Asset>(&self, obj: T) -> RRef<T> { RRef::new(obj, self.uuid, self.sys.deleter.clone()) }
}

pub trait AssetRuntime: 'static + Send + Sync {
	fn initialize(device: &Device) -> radiance_graph::Result<Self>
	where
		Self: Sized;

	fn as_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

pub trait Asset: Sized {
	const TYPE: Uuid;
	const MODIFIABLE: bool;
	type Import;
	type Runtime: AssetRuntime;

	fn initialize(ctx: InitContext<'_, Self::Runtime>) -> LResult<Self>;

	fn write(&self, into: Writer) -> Result<(), std::io::Error>;

	fn import(name: &str, import: Self::Import, into: Writer) -> Result<(), std::io::Error>;

	fn into_resources(self, queue: Sender<DelRes>);
}

pub trait ImportSettings {
	fn has_settings(&self) -> bool;
}

pub trait Importer {
	type Settings: ImportSettings;

	fn initialize(path: &Path) -> Option<Result<Self, std::io::Error>>
	where
		Self: Sized;

	fn settings(&mut self) -> &mut Self::Settings;

	fn import(self, into: AssetSystemView, progress: impl Fn(f32) + Send + Sync) -> Result<(), std::io::Error>;
}

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

pub struct AssetSystem {
	deleter: Sender<DelRes>,
	delete_recv: Receiver<DelRes>,
	loaded: FxDashMap<Uuid, RWeak>,
	path_lookup: FxDashMap<Uuid, PathBuf>,
	runtimes: FxHashMap<Uuid, Arc<dyn AssetRuntime>>,
	root: Dir,
}

impl AssetSystem {
	pub fn new(device: &Device, path: &Path) -> radiance_graph::Result<Self> {
		let root = Dir {
			path: path.to_path_buf(),
			sub: FxDashMap::default(),
		};
		let path_lookup = DashMap::default();
		for item in WalkDir::new(path).into_iter().filter_map(|x| x.ok()) {
			let fp = item.path();
			let p = fp.strip_prefix(path).unwrap_or(item.path());
			if fp.is_file() {
				let Ok(header) = AssetHeader::from_path(fp) else {
					continue;
				};
				root.add(p, Some(header));
				path_lookup.insert(header.asset, fp.to_path_buf());
			} else {
				root.add(p, None);
			}
		}

		let (send, recv) = crossbeam_channel::unbounded();
		let mut this = Self {
			deleter: send,
			delete_recv: recv,
			loaded: DashMap::default(),
			path_lookup,
			runtimes: FxHashMap::default(),
			root,
		};
		this.register::<mesh::Mesh>(device)?;
		this.register::<scene::Scene>(device)?;
		Ok(this)
	}

	pub fn register<T: Asset>(&mut self, device: &Device) -> radiance_graph::Result<()> {
		self.runtimes.insert(T::TYPE, Arc::new(T::Runtime::initialize(device)?));
		Ok(())
	}

	pub fn view(&self, cursor: PathBuf) -> AssetSystemView { AssetSystemView { sys: self, cursor } }

	pub fn initialize<T: Asset>(&self, device: &Device, uuid: Uuid) -> Result<RRef<T>, LoadError> {
		match self.loaded.entry(uuid) {
			Entry::Occupied(mut o) => match o.get().clone().upgrade() {
				Some(x) => Ok(x),
				None => {
					let r = self.load_from_disk::<T>(device, uuid)?;
					o.insert(r.downgrade());
					Ok(r)
				},
			},
			Entry::Vacant(v) => {
				let r = self.load_from_disk::<T>(device, uuid)?;
				v.insert(r.downgrade());
				Ok(r)
			},
		}
	}

	pub fn save<T: Asset>(&self, asset: &RRef<T>) -> Result<(), std::io::Error> {
		assert!(T::MODIFIABLE, "Can only save modifiable assets");

		let path = self.path_lookup.get(&asset.uuid()).unwrap();
		let mut file = OpenOptions::new().read(true).write(true).open(&*path)?;
		let header = AssetHeader::from_file(&mut file)?;
		if header.ty != T::TYPE {
			return Err(std::io::Error::other("asset type mismatch").into());
		}
		asset.write(Writer::from_file(file)?)
	}

	fn load_from_disk<T: Asset>(&self, device: &Device, uuid: Uuid) -> Result<RRef<T>, LoadError> {
		let path = self.path_lookup.get(&uuid).unwrap();
		let mut file = File::open(&*path)?;
		let header = AssetHeader::from_file(&mut file)?;
		if header.ty != T::TYPE {
			return Err(std::io::Error::other("asset type mismatch").into());
		}

		T::initialize(InitContext {
			name: path.file_name().unwrap().to_str().unwrap(),
			device,
			uuid: header.asset,
			data: Reader::from_file(file)?,
			sys: self,
			runtime: Arc::downcast(
				self.runtimes
					.get(&T::TYPE)
					.expect("asset has not been registered")
					.clone()
					.as_any(),
			)
			.unwrap(),
		})
	}

	pub fn tick(&self, frame: &mut Frame) {
		while let Ok(x) = self.delete_recv.try_recv() {
			match x {
				DelRes::Resource(x) => frame.delete(x),
			}
		}
	}

	pub unsafe fn destroy(self, device: &Device) {
		for (_, a) in self.loaded {
			assert!(a.is_dead(), "Cannot destroy `AssetSystem` with currently loaded assets")
		}

		for x in self.delete_recv.try_iter() {
			match x {
				DelRes::Resource(r) => unsafe { r.destroy(device) },
			}
		}
	}
}

pub enum LoadError {
	Vulkan(radiance_graph::Error),
	Io(std::io::Error),
}
impl From<std::io::Error> for LoadError {
	fn from(value: std::io::Error) -> Self { Self::Io(value) }
}
impl From<radiance_graph::Error> for LoadError {
	fn from(value: radiance_graph::Error) -> Self { Self::Vulkan(value) }
}
impl Display for LoadError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Vulkan(e) => Debug::fmt(e, f),
			Self::Io(e) => Display::fmt(e, f),
		}
	}
}
impl Debug for LoadError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Display::fmt(self, f) }
}
impl Error for LoadError {}

#[derive(Clone, Debug)]
enum DirEntry {
	Dir(Dir),
	Asset(AssetHeader),
}

#[derive(Clone, Debug)]
struct Dir {
	path: PathBuf,
	sub: FxDashMap<PathBuf, DirEntry>,
}

impl Dir {
	fn add(&self, path: &Path, header: Option<AssetHeader>) {
		let mut comp = path.components();
		if let Some(c) = comp.next() {
			let p: &Path = c.as_ref();
			if self.path.join(p).is_file() {
				self.sub.insert(p.to_path_buf(), DirEntry::Asset(header.unwrap()));
			} else {
				let sub_dir = self.sub.entry(p.to_path_buf()).or_insert_with(|| {
					DirEntry::Dir(Dir {
						path: self.path.join(p),
						sub: FxDashMap::default(),
					})
				});
				match *sub_dir {
					DirEntry::Dir(ref d) => d.add(comp.as_path(), header),
					_ => unreachable!("tried to add subdir to asset"),
				}
			}
		}
	}

	fn remove(&self, path: &Path) -> DirEntry {
		let mut comp = path.components();
		if let Some(c) = comp.next() {
			let p: &Path = c.as_ref();
			let rest = comp.as_path();
			if rest.as_os_str().is_empty() {
				self.sub.remove(p).unwrap().1
			} else {
				let child = self.sub.get(p).unwrap();
				match child.value() {
					DirEntry::Dir(d) => d.remove(rest),
					_ => unreachable!("tried to remove subdir of asset"),
				}
			}
		} else {
			unreachable!("cannot remove root")
		}
	}

	fn move_into<'a>(&self, into: &Path, from: impl Iterator<Item = &'a Path>) -> Vec<(PathBuf, DirEntry)> {
		let entries: Vec<_> = from
			.map(|x| (PathBuf::from(x.file_name().unwrap()), self.remove(x)))
			.collect();
		self.move_into_inner(into, &entries);
		entries
	}

	fn move_into_inner(&self, into: &Path, entries: &[(PathBuf, DirEntry)]) {
		let mut comp = into.components();
		if let Some(c) = comp.next() {
			let p: &Path = c.as_ref();
			let rest = comp.as_path();
			let child = self.sub.get(p).unwrap();
			match child.value() {
				DirEntry::Dir(ref d) => d.move_into_inner(rest, entries),
				_ => unreachable!("tried to get subdir of asset"),
			}
		} else {
			for (p, e) in entries {
				self.sub.insert(p.clone(), e.clone());
			}
		}
	}

	fn get(&self, path: &Path) -> Vec<(String, DirView)> {
		let mut comp = path.components();
		if let Some(c) = comp.next() {
			let p: &Path = c.as_ref();
			let rest = comp.as_path();
			let child = self.sub.get(p);
			child
				.map(|x| match x.value() {
					DirEntry::Dir(ref d) => d.get(rest),
					_ => unreachable!("tried to get subdir of asset"),
				})
				.unwrap_or(Vec::new())
		} else {
			self.sub
				.iter()
				.map(|x| {
					(
						x.key().to_str().unwrap().to_string(),
						match *x.value() {
							DirEntry::Dir(_) => DirView::Dir,
							DirEntry::Asset(h) => DirView::Asset(h),
						},
					)
				})
				.collect()
		}
	}
}

pub struct AssetSystemView<'a> {
	sys: &'a AssetSystem,
	cursor: PathBuf,
}

pub enum DirView {
	Dir,
	Asset(AssetHeader),
}

impl AssetSystemView<'_> {
	pub fn create_dir(&self, name: &str) -> Result<(), std::io::Error> {
		let full_path = self.cursor.join(name);
		self.sys.root.add(&full_path, None);
		std::fs::create_dir(self.sys.root.path.join(full_path))
	}

	pub fn delete(&self, name: &str) -> Result<(), std::io::Error> {
		let full_path = self.cursor.join(name);
		self.sys.root.remove(&full_path);
		let path = self.sys.root.path.join(full_path);
		if path.is_file() {
			std::fs::remove_file(path)
		} else {
			std::fs::remove_dir_all(path)
		}
	}

	pub fn move_into<'a>(&self, into: &str, from: impl Iterator<Item = &'a str>) -> Result<(), std::io::Error> {
		let mut full_into = self.cursor.clone();
		let as_path = Path::new(into);
		for c in as_path.components() {
			match c {
				Component::ParentDir => {
					full_into.pop();
				},
				x => full_into.push(x),
			}
		}
		let full_froms: Vec<_> = from.map(|x| self.cursor.join(x)).collect();
		let path_to_entry = self
			.sys
			.root
			.move_into(&full_into, full_froms.iter().map(|x| x.as_path()));
		let dest = self.sys.root.path.join(full_into);
		for (p, e) in path_to_entry {
			match e {
				DirEntry::Asset(h) => {
					self.sys.path_lookup.insert(h.asset, dest.join(p));
				},
				_ => {},
			}
		}
		for from in full_froms {
			let dest = dest.join(from.file_name().unwrap());
			let src = self.sys.root.path.join(from);
			let _ = std::fs::rename(src, dest);
		}

		Ok(())
	}

	pub fn entries(&self) -> impl Iterator<Item = (String, DirView)> + '_ {
		self.sys.root.get(&self.cursor).into_iter()
	}

	pub fn writer(&self, path: &Path, header: AssetHeader) -> Result<Writer, std::io::Error> {
		let full_path = self.cursor.join(path);
		let fs_path = self.sys.root.path.join(&full_path);
		let w = Writer::from_path(&fs_path, header);
		self.sys.root.add(&full_path, Some(header));
		self.sys.path_lookup.insert(header.asset, fs_path);
		w
	}
}
