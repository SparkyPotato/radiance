use std::{
	mem::MaybeUninit,
	ops::Deref,
	sync::{Arc, Weak},
};

use crossbeam_channel::Sender;

use crate::DelRes;

pub trait RuntimeAsset {
	fn into_resources(self, queue: Sender<DelRes>);
}

pub struct RRef<T: RuntimeAsset> {
	inner: Arc<Control<T>>,
}

impl<T: RuntimeAsset> Clone for RRef<T> {
	fn clone(&self) -> Self {
		Self {
			inner: self.inner.clone(),
		}
	}
}

pub struct RWeak<T: RuntimeAsset> {
	inner: Weak<Control<T>>,
}

impl<T: RuntimeAsset> Clone for RWeak<T> {
	fn clone(&self) -> Self {
		Self {
			inner: self.inner.clone(),
		}
	}
}

struct Control<T: RuntimeAsset> {
	obj: MaybeUninit<T>,
	deleter: MaybeUninit<Sender<DelRes>>,
}

impl<T: RuntimeAsset> RRef<T> {
	pub fn new(obj: T, deleter: Sender<DelRes>) -> Self {
		let c = Control {
			obj: MaybeUninit::new(obj),
			deleter: MaybeUninit::new(deleter),
		};
		Self { inner: Arc::new(c) }
	}

	pub fn downgrade(&self) -> RWeak<T> {
		let inner = Arc::downgrade(&self.inner);
		RWeak { inner }
	}

	pub fn ptr_eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.inner, &other.inner) }
}

impl<T: RuntimeAsset> RWeak<T> {
	pub fn new() -> Self { Self { inner: Weak::new() } }

	pub fn upgrade(&self) -> Option<RRef<T>> { self.inner.upgrade().map(|inner| RRef { inner }) }

	pub fn ptr_eq(&self, other: &Self) -> bool { Weak::ptr_eq(&self.inner, &other.inner) }
}

impl<T: RuntimeAsset> Deref for RRef<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target { unsafe { self.inner.as_ref().obj.assume_init_ref() } }
}

impl<T: RuntimeAsset> Drop for Control<T> {
	fn drop(&mut self) {
		unsafe {
			let obj = self.obj.assume_init_read();
			let deleter = self.deleter.assume_init_read();
			obj.into_resources(deleter);
		}
	}
}

