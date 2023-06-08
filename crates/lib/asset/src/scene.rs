use bytemuck::bytes_of;
use uuid::Uuid;
use vek::Mat4;

use crate::util::{SliceReader, SliceWriter};

pub struct Node {
	pub name: String,
	pub transform: Mat4<f32>,
	pub model: Uuid,
}

impl Node {
	fn size(&self) -> usize {
		std::mem::size_of::<u32>() + self.name.len() + std::mem::size_of::<Mat4<f32>>() + std::mem::size_of::<Uuid>()
	}

	/// - 1 u32: length of name.
	/// - name.
	/// - 16 bytes: transform.
	/// - 16 bytes: model UUID.
	fn write(&self, writer: &mut SliceWriter) {
		writer.write(self.name.len() as u32);
		writer.write_slice(self.name.as_bytes());
		writer.write_slice(bytes_of(&self.transform.cols));
		writer.write_slice(self.model.as_bytes());
	}

	fn read(reader: &mut SliceReader) -> Self {
		let name_len = reader.read::<u32>() as usize;
		let name = String::from_utf8(reader.read_slice(name_len).to_vec()).unwrap();
		let transform = Mat4::from_col_array(reader.read_slice(16).try_into().unwrap());
		let model = Uuid::from_slice(reader.read_slice(16)).unwrap();

		Self { name, transform, model }
	}
}

pub struct Scene {
	pub nodes: Vec<Node>,
}

impl Scene {
	/// - 1 u32: node count.
	/// - nodes.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut bytes = vec![0; std::mem::size_of::<u32>() + self.nodes.iter().map(|x| x.size()).sum::<usize>()];
		let mut writer = SliceWriter::new(&mut bytes);

		writer.write(self.nodes.len() as u32);
		for node in &self.nodes {
			node.write(&mut writer);
		}

		zstd::encode_all(bytes.as_slice(), 8).unwrap()
	}

	pub fn from_bytes(bytes: &[u8]) -> Self {
		let bytes = zstd::decode_all(bytes).unwrap();
		let mut reader = SliceReader::new(&bytes);

		let node_count = reader.read::<u32>() as usize;
		let mut nodes = Vec::with_capacity(node_count);
		for _ in 0..node_count {
			nodes.push(Node::read(&mut reader));
		}

		Self { nodes }
	}
}
