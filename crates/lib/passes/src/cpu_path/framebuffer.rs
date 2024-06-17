use std::{cell::UnsafeCell, mem, ops::Range, ptr, slice, sync::Mutex};

use rayon::prelude::*;
use vek::{Rgba, Vec2};

const TILE_SIZE: u32 = 16;

pub struct Tile<'a> {
	pixel: Vec2<u32>,
	pub offset: Vec2<u32>,
	size: Vec2<u32>,
	pixels: slice::IterMut<'a, Rgba<f16>>,
}

pub struct RenderPixel<'a> {
	pub pixel: Vec2<u32>,
	pub data: &'a mut Rgba<f16>,
}

impl<'a> Iterator for Tile<'a> {
	type Item = RenderPixel<'a>;

	fn next(&mut self) -> Option<Self::Item> {
		let data = self.pixels.next()?;
		let pixel = self.pixel;

		self.pixel.x += 1;
		if self.pixel.x == self.offset.x + self.size.x {
			self.pixel.x = 0;
			self.pixel.y += 1;
		}

		Some(RenderPixel { pixel, data })
	}
}

#[derive(Clone, Debug)]
struct TileData {
	offset: Vec2<u32>,
	size: Vec2<u32>,
	data: Range<usize>,
}

pub struct Framebuffer {
	size: Vec2<u32>,
	data: UnsafeCell<Vec<Rgba<f16>>>,
	pub read: Mutex<Vec<u8>>,
	tiles: Vec<TileData>,
}

unsafe impl Sync for Framebuffer {}

impl Framebuffer {
	pub fn new(size: Vec2<u32>) -> Self {
		let tiles: Vec<_> = (0..((size.x + TILE_SIZE - 1) / TILE_SIZE))
			.flat_map(|xt| (0..((size.y + TILE_SIZE - 1) / TILE_SIZE)).map(move |yt| (xt, yt)))
			.map(|(xt, yt)| {
				let offset = Vec2::new(xt, yt) * TILE_SIZE;
				let fsize = size;
				let size = (size - offset).map(|x| x.min(TILE_SIZE));
				let start = offset.y as usize * fsize.x as usize + offset.x as usize * size.y as usize;
				TileData {
					offset,
					size,
					data: start..(start + size.x as usize * size.y as usize),
				}
			})
			.collect();
		let plen = size.x as usize * size.y as usize;
		Self {
			size,
			data: UnsafeCell::new(vec![Rgba::new(0.0, 0.0, 0.0, 0.0); plen]),
			read: Mutex::new(vec![0; plen * std::mem::size_of::<Rgba<f16>>()]),
			tiles,
		}
	}

	pub fn size(&self) -> Vec2<u32> { self.size }

	pub fn present(&self) {
		unsafe {
			let mut to = self.read.lock().unwrap();
			let from = &*self.data.get();
			for tile in self.tiles.iter() {
				let mut v = (to.as_mut_ptr() as *mut Rgba<f16>)
					.add(tile.offset.y as usize * self.size.x as usize + tile.offset.x as usize);
				for row in from[tile.data.clone()].chunks(tile.size.x as _) {
					ptr::copy_nonoverlapping(row.as_ptr(), v, row.len());
					v = v.add(self.size.x as usize)
				}
			}
		}

		// unsafe {
		// 	let v = &*self.data.get();
		// 	self.read
		// 		.lock()
		// 		.unwrap()
		// 		.copy_from_slice(slice::from_raw_parts(v.as_ptr() as _, v.len() * 8));
		// }
	}

	pub fn get_tiles(&self) -> impl ParallelIterator<Item = Tile> {
		self.tiles
			.iter()
			.map(|data| Tile {
				pixel: data.offset,
				offset: data.offset,
				size: data.size,
				pixels: unsafe {
					slice::from_raw_parts_mut((*self.data.get()).as_ptr().add(data.data.start) as _, data.data.len())
						.iter_mut()
				},
			})
			.par_bridge()
	}
}
