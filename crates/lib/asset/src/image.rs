use bincode::{Decode, Encode};

#[derive(Copy, Clone, Eq, PartialEq, Encode, Decode)]
pub enum Format {
	R8,
	R8G8,
	R8G8B8A8,
	R16,
	R16G16,
	R16G16B16,
	R16G16B16A16,
	R32G32B32FLOAT,
	R32G32B32A32FLOAT,
}

#[derive(Encode, Decode)]
pub struct Image {
	pub width: u32,
	pub height: u32,
	pub format: Format,
	pub data: Vec<u8>,
}

