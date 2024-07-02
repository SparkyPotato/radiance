//! Fast arena allocator for ephemeral data that lasts the lifetime of a frame.

use std::{
	alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout},
	borrow::Borrow,
	cell::UnsafeCell,
	collections::BTreeMap,
	ptr::{addr_of_mut, NonNull},
};

use tracing::trace;

pub trait FromIteratorAlloc<T, A: Allocator>: Sized {
	fn from_iter_alloc<I: IntoIterator<Item = T>>(iter: I, alloc: A) -> Self;
}

pub trait IteratorAlloc: Iterator {
	fn collect_in<A: Allocator, C: FromIteratorAlloc<Self::Item, A>>(self, alloc: A) -> C;
}

impl<I: Iterator> IteratorAlloc for I {
	fn collect_in<A: Allocator, C: FromIteratorAlloc<Self::Item, A>>(self, alloc: A) -> C {
		C::from_iter_alloc(self, alloc)
	}
}

impl<T, A: Allocator> FromIteratorAlloc<T, A> for Vec<T, A> {
	fn from_iter_alloc<I: IntoIterator<Item = T>>(iter: I, alloc: A) -> Self {
		let iter = iter.into_iter();
		let mut vec = Vec::new_in(alloc);
		vec.extend(iter);
		vec
	}
}

impl<K: Ord, V, A: Allocator + Clone> FromIteratorAlloc<(K, V), A> for BTreeMap<K, V, A> {
	fn from_iter_alloc<I: IntoIterator<Item = (K, V)>>(iter: I, alloc: A) -> Self {
		let iter = iter.into_iter();
		let mut map = BTreeMap::new_in(alloc);
		map.extend(iter);
		map
	}
}

pub trait ToOwnedAlloc<A> {
	type Owned;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned;
}

impl<T: Clone, A: Allocator> ToOwnedAlloc<A> for [T] {
	type Owned = Vec<T, A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned { self.to_vec_in(alloc) }
}

impl<A: Allocator, T: Clone> ToOwnedAlloc<A> for &'_ [T] {
	type Owned = Vec<T, A>;

	fn to_owned_alloc(&self, alloc: A) -> Self::Owned { (**self).to_owned_alloc(alloc) }
}

#[repr(C, align(8))]
struct Block {
	header: BlockHeader,
	data: [u8],
}

#[repr(align(8))]
struct BlockHeader {
	next: Option<NonNull<Block>>,
	offset: usize,
}

pub struct Arena {
	inner: UnsafeCell<Inner>,
}

struct Inner {
	head: NonNull<Block>,
	curr_block: NonNull<Block>,
	alloc_count: usize,
	last_alloc: usize,
}

impl Default for Arena {
	fn default() -> Self { Self::new() }
}

impl Arena {
	/// Creates a new arena with a default block size of 1 MiB.
	pub fn new() -> Self { Self::with_block_size(1024 * 1024) }

	pub fn memory_usage(&self) -> usize {
		unsafe {
			let mut size = 0;
			let inner = self.inner.get();

			let mut block = Some((*inner).head);
			while let Some(mut b) = block {
				size += b.as_mut().header.offset;
				block = b.as_mut().header.next;
			}

			size
		}
	}

	/// Creates a new arena with the given block size in bytes.
	pub fn with_block_size(block_size: usize) -> Self {
		let head = match Self::allocate_block(block_size) {
			Ok(head) => head,
			Err(_) => handle_alloc_error(Self::block_layout(block_size)),
		};

		Arena {
			inner: UnsafeCell::new(Inner {
				head,
				curr_block: head,
				alloc_count: 0,
				last_alloc: 0,
			}),
		}
	}

	/// Confirm that the arena has been reset to the beginning.
	///
	/// Should be called at the start of every frame.
	pub fn reset(&mut self) {
		let count = self.inner.get_mut().alloc_count;
		if count != 0 {
			panic!("tried to reset Arena with living allocations ({})", count);
		}
		unsafe {
			self.reset_all_blocks();
		}
	}

	/// [`Allocator::deallocate`], but doesn't require a layout.
	///
	/// # Safety
	/// Same as `Allocator::deallocate`.
	pub unsafe fn deallocate(&self, _: NonNull<u8>) {
		let inner = self.inner.get();
		(*inner).alloc_count -= 1;
	}

	unsafe fn reset_all_blocks(&self) {
		let inner = self.inner.get();
		let mut block = Some((*inner).head);
		while let Some(b) = block {
			let b = b.as_ptr();
			(*b).header.offset = 0;
			block = (*b).header.next;
		}
		(*inner).curr_block = (*inner).head;
	}

	fn block_layout(size: usize) -> Layout {
		unsafe {
			Layout::from_size_align_unchecked(
				std::mem::size_of::<BlockHeader>() + size,
				std::mem::align_of::<BlockHeader>(),
			)
		}
	}

	fn allocate_block(size: usize) -> Result<NonNull<Block>, AllocError> {
		unsafe {
			let head: NonNull<Block> = Global
				.allocate(Self::block_layout(size))
				.map(|ptr| NonNull::new_unchecked(std::ptr::from_raw_parts_mut(ptr.as_ptr() as *mut (), size)))?;

			addr_of_mut!((*head.as_ptr()).header).write(BlockHeader { next: None, offset: 0 });

			Ok(head)
		}
	}

	fn extend(&self, size: usize) -> Result<NonNull<Block>, AllocError> {
		trace!("arena: block is full, allocating new block");

		let inner = self.inner.get();
		let new = Self::allocate_block(size)?;
		unsafe {
			(*(*inner).curr_block.as_ptr()).header.next = Some(new);
			(*inner).curr_block = new;
		}

		Ok(new)
	}

	fn aligned_offset(&self, align: usize) -> usize {
		unsafe {
			let curr = (*self.inner.get()).curr_block.as_ptr();
			let base = (*curr).data.as_ptr();
			let unaligned = base.add((*curr).header.offset) as usize;
			let aligned = (unaligned + align - 1) & !(align - 1);
			aligned - base as usize
		}
	}
}

unsafe impl Allocator for Arena {
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		// SAFETY: Uh.
		unsafe {
			let inner = self.inner.get();

			let (ptr, offset) = if unlikely(layout.size() > (*(*inner).curr_block.as_ptr()).data.len()) {
				// Allocate a dedicated block for this, since it's too big for our current block size.
				let ptr = addr_of_mut!((*self.extend(layout.size())?.as_ptr()).data).cast();
				(ptr, layout.size())
			} else {
				let mut offset = self.aligned_offset(layout.align());
				if unlikely(offset + layout.size() > (*(*inner).curr_block.as_ptr()).data.len()) {
					// There's not enough space in the current block, so go to the next one.
					if let Some(next) = (*(*inner).curr_block.as_ptr()).header.next {
						// There's a next block, so we can use it.
						(*inner).curr_block = next;
					} else {
						// There's no next block, so we need to allocate a new one.
						self.extend((*(*inner).curr_block.as_ptr()).data.len())?;
					}

					offset = self.aligned_offset(layout.align());
				}

				let target = addr_of_mut!((*(*inner).curr_block.as_ptr()).data)
					.cast::<u8>()
					.add(offset);
				(target, offset + layout.size())
			};

			(*inner).alloc_count += 1;
			(*inner).last_alloc = ptr.to_raw_parts().0.addr();
			(*(*inner).curr_block.as_ptr()).header.offset = offset;

			Ok(NonNull::new_unchecked(std::ptr::from_raw_parts_mut(
				ptr as _,
				layout.size(),
			)))
		}
	}

	unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) { self.deallocate(ptr); }

	unsafe fn grow(
		&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		let inner = self.inner.get();

		if likely(ptr.addr().get() == (*inner).last_alloc) {
			// Reuse the last allocation if possible.
			let offset = ptr.as_ptr().offset_from((*(*inner).curr_block.as_ptr()).data.as_ptr());
			let new_offset = offset as usize + new_layout.size();
			if likely(new_offset <= (*(*inner).curr_block.as_ptr()).data.len()) {
				(*(*inner).curr_block.as_ptr()).header.offset = new_offset;
				return Ok(NonNull::new_unchecked(std::ptr::from_raw_parts_mut(
					ptr.as_ptr() as _,
					new_layout.size(),
				)));
			}
		}

		let new_ptr = self.allocate(new_layout)?;
		std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr() as *mut _, old_layout.size());
		(*inner).alloc_count -= 1;
		Ok(new_ptr)
	}
}

impl Drop for Arena {
	fn drop(&mut self) {
		let inner = self.inner.get_mut();

		let mut block = Some(inner.head);
		while let Some(mut b) = block {
			let mut prev = b;
			unsafe {
				block = b.as_mut().header.next;
				Global.deallocate(prev.cast(), Self::block_layout(prev.as_mut().data.len()));
			}
		}
	}
}

#[inline]
#[cold]
fn cold() {}

#[inline]
fn likely(b: bool) -> bool {
	if !b {
		cold()
	}
	b
}

#[inline]
fn unlikely(b: bool) -> bool {
	if b {
		cold()
	}
	b
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn non_overlapping() {
		let arena = Arena::new();

		unsafe {
			let a = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;
			let b = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;

			*a = 123;
			*b = 456;

			assert_eq!(*a, 123);
			assert_eq!(*b, 456);
		}
	}

	#[test]
	fn allocate_over_size() {
		let arena = Arena::with_block_size(256);

		let _vec = Vec::<u8, &Arena>::with_capacity_in(178, &arena);
		let _vec = Vec::<u8, &Arena>::with_capacity_in(128, &arena);
	}

	#[test]
	#[should_panic]
	fn early_reset() {
		let mut arena = Arena::new();

		let _ = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;
		arena.reset();
	}

	#[test]
	fn grow() {
		let arena = Arena::new();

		unsafe {
			let a = arena.allocate(Layout::new::<u32>()).unwrap().cast();
			let b = arena
				.grow(a, Layout::new::<u32>(), Layout::new::<[u32; 2]>())
				.unwrap()
				.cast();
			assert_eq!(a, b);
		}
	}
}
