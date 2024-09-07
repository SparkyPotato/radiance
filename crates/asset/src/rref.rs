use std::{
	mem::MaybeUninit,
	ops::Deref,
	sync::{Arc, Weak},
};

use crossbeam_channel::Sender;
use radiance_graph::graph::Resource;

use crate::Asset;

pub enum DelRes {
	Resource(Resource),
}

impl From<Resource> for DelRes {
	fn from(value: Resource) -> Self { Self::Resource(value) }
}

pub struct RRef<T: Asset> {
	inner: Arc<Control<T>>,
}

impl<T: Asset> Clone for RRef<T> {
	fn clone(&self) -> Self {
		Self {
			inner: self.inner.clone(),
		}
	}
}

pub struct RWeak {
	inner: Weak<()>,
}

impl Clone for RWeak {
	fn clone(&self) -> Self {
		Self {
			inner: self.inner.clone(),
		}
	}
}

struct Control<T: Asset> {
	deleter: MaybeUninit<Sender<DelRes>>,
	obj: MaybeUninit<T>,
}

impl<T: Asset> RRef<T> {
	pub fn new(obj: T, deleter: Sender<DelRes>) -> Self {
		let c = Control {
			deleter: MaybeUninit::new(deleter),
			obj: MaybeUninit::new(obj),
		};
		Self { inner: Arc::new(c) }
	}

	pub fn downgrade(&self) -> RWeak {
		let inner = Arc::downgrade(&self.inner);
		RWeak {
			inner: unsafe { Weak::from_raw(Weak::into_raw(inner) as _) },
		}
	}

	pub fn ptr_eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.inner, &other.inner) }
}

impl RWeak {
	pub fn new() -> Self { Self { inner: Weak::new() } }

	pub fn is_dead(&self) -> bool { self.inner.strong_count() == 0 }

	pub fn upgrade<T: Asset>(self) -> Option<RRef<T>> {
		let inner = unsafe { Weak::from_raw(Weak::into_raw(self.inner) as _) };
		inner.upgrade().map(|inner| RRef { inner })
	}

	pub fn ptr_eq(&self, other: &Self) -> bool { Weak::ptr_eq(&self.inner, &other.inner) }
}

impl<T: Asset> Deref for RRef<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target { unsafe { self.inner.as_ref().obj.assume_init_ref() } }
}

impl<T: Asset> Drop for Control<T> {
	fn drop(&mut self) {
		unsafe {
			let obj = self.obj.assume_init_read();
			let deleter = self.deleter.assume_init_read();
			obj.into_resources(deleter);
		}
	}
}
