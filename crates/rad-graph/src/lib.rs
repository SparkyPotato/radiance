#![feature(allocator_api)]
#![feature(btreemap_alloc)]
#![feature(ptr_metadata)]
#![feature(slice_ptr_get)]

use std::fmt::{Debug, Display};

pub use ash;
pub use gpu_allocator::{vulkan as alloc, MemoryLocation};

pub mod arena;
pub mod cmd;
pub mod device;
pub mod graph;
pub mod resource;
pub mod sync;
pub mod util;

#[derive(Clone)]
pub enum Error {
	Message(String),
	Vulkan(ash::vk::Result),
}

impl Display for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Error::Message(msg) => write!(f, "{}", msg),
			Error::Vulkan(res) => write!(f, "Vulkan error: {}", res),
		}
	}
}

impl Debug for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { <Self as Display>::fmt(self, f) }
}

impl From<String> for Error {
	fn from(message: String) -> Self { Error::Message(message) }
}

impl From<ash::vk::Result> for Error {
	fn from(result: ash::vk::Result) -> Self { Error::Vulkan(result) }
}

pub type Result<T> = std::result::Result<T, Error>;
