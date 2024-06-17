use std::{cell::UnsafeCell, sync::Mutex};

use bytemuck::cast_slice;
use half::f16;
use rayon::prelude::*;
use vek::{Rgba, Vec2};

const TILE_SIZE: u32 = 16;

pub struct Tile<'a> {
	end: Vec2<u32>,
	pixel: Vec2<u32>,
	frame: &'a Framebuffer,
}

pub struct RenderPixel<'a> {
	pub pixel: Vec2<u32>,
	pub data: &'a mut Rgba<f16>,
}

impl<'a> Iterator for Tile<'a> {
	type Item = RenderPixel<'a>;

	fn next(&mut self) -> Option<Self::Item> {
		let pixel = self.pixel;
		if pixel.y == self.end.y {
			None
		} else {
			self.pixel.x += 1;
			if self.pixel.x == self.end.x {
				self.pixel.x = 0;
				self.pixel.y += 1;
			}
			Some(RenderPixel {
				pixel,
				data: unsafe {
					&mut *((*self.frame.data.get())
						.as_ptr()
						.add(pixel.y as usize * self.frame.size.x as usize + pixel.x as usize) as *mut _)
				},
			})
		}
	}
}

#[derive(Clone, Debug)]
struct TileData {
	offset: Vec2<u32>,
	size: Vec2<u32>,
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
				let size = (size - offset).map(|x| x.min(TILE_SIZE));
				TileData { offset, size }
			})
			.collect();
		let res = size.x as usize * size.y as usize;

		Self {
			size,
			data: UnsafeCell::new(vec![Rgba::zero(); res]),
			read: Mutex::new(vec![0; res * std::mem::size_of::<Rgba<f16>>()]),
			tiles,
		}
	}

	pub fn size(&self) -> Vec2<u32> { self.size }

	pub fn present(&self) {
		unsafe {
			let v = &mut *self.data.get();
			self.read.lock().unwrap().copy_from_slice(cast_slice(&v));
		}
	}

	pub fn get_tiles(&self) -> impl ParallelIterator<Item = Tile> {
		self.tiles
			.iter()
			.map(|data| Tile {
				end: data.offset + data.size,
				pixel: data.offset,
				frame: self,
			})
			.par_bridge()
	}
}
