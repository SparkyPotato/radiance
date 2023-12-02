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
		let extra = self.name.len() % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		std::mem::size_of::<u32>()
			+ self.name.len()
			+ std::mem::size_of::<Mat4<f32>>()
			+ std::mem::size_of::<Uuid>()
			+ fill
	}

	/// - 16 bytes: transform.
	/// - 16 bytes: model UUID.
	/// - 1 u32: length of name.
	/// - name.
	fn write(&self, writer: &mut SliceWriter) {
		writer.write_slice(bytes_of(&self.transform.cols)).unwrap();
		writer.write_slice(self.model.as_bytes()).unwrap();
		writer.write(self.name.len() as u32).unwrap();
		writer.write_slice(self.name.as_bytes()).unwrap();
		let extra = self.name.len() % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		writer.write_slice(&[0u8, 0, 0][0..fill]).unwrap();
	}

	fn read(reader: &mut SliceReader) -> Result<Self, ()> {
		let transform = Mat4::from_col_array(reader.read_slice(16).ok_or(())?.try_into().unwrap());
		let model = Uuid::from_slice(reader.read_slice(16).ok_or(())?).unwrap();
		let name_len = reader.read::<u32>().ok_or(())? as usize;
		let name = String::from_utf8(reader.read_slice(name_len).ok_or(())?.to_vec()).unwrap();
		let extra = name_len % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		reader.read_slice::<u8>(fill);

		Ok(Self { name, transform, model })
	}
}

pub enum Projection {
	Perspective { yfov: f32, near: f32, far: Option<f32> },
	Orthographic { height: f32, near: f32, far: f32 },
}

pub struct Camera {
	pub name: String,
	pub view: Mat4<f32>,
	pub projection: Projection,
}

impl Camera {
	fn size(&self) -> usize {
		let extra = self.name.len() % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		std::mem::size_of::<Mat4<f32>>()
			+ std::mem::size_of::<u32>()
			+ std::mem::size_of::<f32>() * 3
			+ std::mem::size_of::<u32>()
			+ self.name.len()
			+ fill
	}

	/// - 64 bytes: transform.
	/// - 1 u32: type of projection.
	/// - 3 f32: projection.
	/// - 1 u32: length of name.
	/// - name.
	fn write(&self, writer: &mut SliceWriter) {
		writer.write(self.view.cols).unwrap();
		match self.projection {
			Projection::Perspective { yfov, near, far } => {
				writer.write(0u32).unwrap();
				writer.write(yfov).unwrap();
				writer.write(near).unwrap();
				writer.write(far.unwrap_or(f32::NAN)).unwrap();
			},
			Projection::Orthographic { height, near, far } => {
				writer.write(1u32).unwrap();
				writer.write(height).unwrap();
				writer.write(near).unwrap();
				writer.write(far).unwrap();
			},
		}
		writer.write(self.name.len() as u32).unwrap();
		writer.write_slice(self.name.as_bytes()).unwrap();
		let extra = self.name.len() % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		writer.write_slice(&[0u8, 0, 0][0..fill]).unwrap();
	}

	fn read(reader: &mut SliceReader) -> Result<Self, ()> {
		let view = Mat4::from_col_array(reader.read_slice(16).ok_or(())?.try_into().unwrap());
		let ty = reader.read::<u32>().ok_or(())?;
		let projection = match ty {
			0 => {
				let yfov = reader.read().ok_or(())?;
				let near = reader.read().ok_or(())?;
				let far = reader.read::<f32>().ok_or(())?;
				Projection::Perspective {
					yfov,
					near,
					far: if far.is_nan() { None } else { Some(far) },
				}
			},
			1 => {
				let height = reader.read().ok_or(())?;
				let near = reader.read().ok_or(())?;
				let far = reader.read().ok_or(())?;
				Projection::Orthographic { height, near, far }
			},
			_ => unreachable!("invalid asset"),
		};
		let name_len = reader.read::<u32>().ok_or(())? as usize;
		let name = String::from_utf8(reader.read_slice(name_len).ok_or(())?.to_vec()).unwrap();
		let extra = name_len % 4;
		let fill = if extra == 0 { 0 } else { 4 - extra };
		reader.read_slice::<u8>(fill);

		Ok(Self { name, view, projection })
	}
}

pub struct Scene {
	pub nodes: Vec<Node>,
	pub cameras: Vec<Camera>,
}

impl Scene {
	/// - 1 u32: node count.
	/// - nodes.
	/// - 1 u32: camera count.
	/// - cameras.
	///
	/// Compressed by zstd.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut bytes = vec![
			0;
			std::mem::size_of::<u32>() * 2
				+ self.nodes.iter().map(|x| x.size()).sum::<usize>()
				+ self.cameras.iter().map(|x| x.size()).sum::<usize>()
		];
		let mut writer = SliceWriter::new(&mut bytes);

		writer.write(self.nodes.len() as u32).unwrap();
		for node in &self.nodes {
			node.write(&mut writer);
		}

		writer.write(self.cameras.len() as u32).unwrap();
		for camera in &self.cameras {
			camera.write(&mut writer);
		}

		zstd::encode_all(bytes.as_slice(), 8).unwrap()
	}

	pub fn from_bytes(bytes: &[u8]) -> Result<Self, ()> {
		let bytes = zstd::decode_all(bytes).unwrap();
		let mut reader = SliceReader::new(&bytes);

		let node_count = reader.read::<u32>().ok_or(())? as usize;
		let mut nodes = Vec::with_capacity(node_count);
		for _ in 0..node_count {
			nodes.push(Node::read(&mut reader)?);
		}

		let camera_count = reader.read::<u32>().ok_or(())? as usize;
		let mut cameras = Vec::with_capacity(camera_count);
		for _ in 0..camera_count {
			cameras.push(Camera::read(&mut reader)?);
		}

		Ok(Self { nodes, cameras })
	}
}
