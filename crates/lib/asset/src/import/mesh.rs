use std::{collections::BTreeMap, ops::Range};

use bytemuck::from_bytes;
use gltf::accessor::{DataType, Dimensions};
use meshopt::{SimplifyOptions, VertexDataAdapter};
use metis::Graph;
use rustc_hash::FxHashMap;
use tracing::{span, Level};
use uuid::Uuid;
use vek::{Sphere, Vec2, Vec3, Vec4};

use crate::{
	import::{ImportError, ImportResult, Importer},
	mesh::{MaterialRange, Mesh, Meshlet, Vertex},
};

impl Importer<'_> {
	pub fn mesh(&self, name: &str, mesh: gltf::Mesh, materials: &[Uuid]) -> ImportResult<Mesh> {
		let s = span!(Level::INFO, "importing mesh", name = name);
		let _e = s.enter();

		let mesh = self.conv_to_mesh(mesh, materials)?;
		let mut out = Mesh {
			vertices: Vec::with_capacity(mesh.vertices.len()),
			indices: Vec::with_capacity(mesh.indices.len()),
			meshlets: Vec::with_capacity(mesh.vertices.len() / 64),
			material_ranges: Vec::with_capacity(mesh.vertices.len() / 64),
		};
		let scale = self.generate_meshlets(&mesh, None, &mut out);

		let mut simplify = 0..out.meshlets.len();
		let mut lod = 1;
		while simplify.len() > 1 {
			let s = span!(Level::TRACE, "generating lod", lod);
			let _e = s.enter();

			let next_start = out.meshlets.len();

			for group in self
				.generate_groups(simplify.clone(), &out)
				.into_iter()
				.filter(|x| x.len() > 1)
			{
				let Some((mesh, group_bounding)) = self.simplify_group(&group, lod, scale, &out) else {
					continue;
				};
				for mid in group {
					out.meshlets[mid].parent_group_bounding = group_bounding;
				}
				self.generate_meshlets(&mesh, Some(group_bounding), &mut out);
			}

			simplify = next_start..out.meshlets.len();
			lod += 1;
		}

		Ok(out)
	}

	fn generate_meshlets(&self, mesh: &FullMesh, group_bounding: Option<Sphere<f32, f32>>, out: &mut Mesh) -> f32 {
		let s = span!(Level::TRACE, "building meshlets");
		let _e = s.enter();

		let adapter = VertexDataAdapter::new(
			bytemuck::cast_slice(mesh.vertices.as_slice()),
			std::mem::size_of::<Vertex>(),
			0,
		)
		.unwrap();
		let meshlets = meshopt::build_meshlets(&mesh.indices, &adapter, 64, 124, 0.5);

		out.meshlets.extend(meshlets.iter().map(|m| {
			let mut vertices = m.vertices.to_vec();
			let mut indices: Vec<_> = m.triangles.iter().map(|&x| x as u32).collect();
			let m_range = sort_by_material(&mut vertices, &mut indices, |x| mesh.get_material_of_vertex(x))
				.into_iter()
				.map(|x| MaterialRange {
					material: x.material,
					vertices: (x.vertices.start as _)..(x.vertices.end as _),
				});

			let vertices = vertices.iter().map(|&x| mesh.vertices[x as usize]);

			let index_offset = out.indices.len() as u32;
			let vertex_offset = out.vertices.len() as u32;
			let vert_count = vertices.len() as u8;
			let tri_count = (indices.len() / 3) as u8;

			out.vertices.extend(vertices);
			out.indices.extend(indices.into_iter().map(|x| x as u8));
			let start = out.material_ranges.len() as _;
			out.material_ranges.extend(m_range);

			let bounds = meshopt::compute_meshlet_bounds(m, &adapter);
			let center = Vec3::from(bounds.center);

			let group_bounding = group_bounding.unwrap_or(Sphere { center, radius: 0.0 });
			Meshlet {
				index_offset,
				vertex_offset,
				tri_count,
				vert_count,
				material_ranges: start..(out.material_ranges.len() as _),
				bounding: Sphere {
					center,
					radius: bounds.radius,
				},
				group_bounding,
				parent_group_bounding: Sphere {
					center: group_bounding.center,
					radius: f32::INFINITY,
				},
			}
		}));

		meshopt::simplify_scale(&adapter)
	}

	fn simplify_group(
		&self, meshlets: &[usize], lod: u32, scale: f32, mesh: &Mesh,
	) -> Option<(FullMesh, Sphere<f32, f32>)> {
		let s = span!(Level::TRACE, "simplifying group");
		let _e = s.enter();
		// TODO: improve simplification.

		let mut indices = Vec::new();
		for &mid in meshlets {
			let m = &mesh.meshlets[mid];
			for i in m.index_offset..(m.index_offset + m.tri_count as u32 * 3) {
				let i = mesh.indices[i as usize] as u32 + m.vertex_offset;
				indices.push(i);
			}
		}

		let t = (lod - 1) as f32 / 20.0;
		let target_error = (0.1 * t + 0.01 * (1.0 - t)) * scale;

		let mut error = 0.0;
		let adapter = VertexDataAdapter::new(
			bytemuck::cast_slice(mesh.vertices.as_slice()),
			std::mem::size_of::<Vertex>(),
			0,
		)
		.unwrap();
		let simplified = meshopt::simplify(
			&indices,
			&adapter,
			indices.len() / 2,
			target_error,
			SimplifyOptions::LockBorder | SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute,
			Some(&mut error),
		);
		error /= 2.0;
		let factor = (simplified.len() / indices.len()) as f32;

		error += meshlets
			.iter()
			.fold(error, |acc, &x| acc.max(mesh.meshlets[x].group_bounding.radius));

		let adapter = VertexDataAdapter::new(
			bytemuck::cast_slice(mesh.vertices.as_slice()),
			std::mem::size_of::<Vertex>(),
			0,
		)
		.unwrap();
		let bounds = meshopt::compute_cluster_bounds(&indices, &adapter);
		let group_bounding = Sphere {
			center: Vec3::from(bounds.center),
			radius: error,
		};

		let mesh = self.make_group_mesh(meshlets, simplified, mesh);

		(factor < 0.75).then_some((mesh, group_bounding))
	}

	fn make_group_mesh(&self, meshlets: &[usize], mut indices: Vec<u32>, mesh: &Mesh) -> FullMesh {
		let mut vertex_map = FxHashMap::default();
		for (i, &v) in indices.iter().enumerate() {
			vertex_map.entry(v).or_insert(Vec::new()).push(i);
		}

		let mut i = 0;
		let mut vertices = Vec::with_capacity(vertex_map.len());
		for (v, is) in vertex_map {
			for id in is {
				indices[id] = i;
			}
			vertices.push(v);

			i += 1;
		}

		FullMesh {
			material_ranges: sort_by_material(&mut vertices, &mut indices, |x| {
				for &mid in meshlets {
					let m = &mesh.meshlets[mid];
					if (m.vertex_offset..(m.vertex_offset + m.vert_count as u32)).contains(&x) {
						let lv = (x - m.vertex_offset) as u8;
						for rid in m.material_ranges.clone() {
							let r = &mesh.material_ranges[rid as usize];
							if r.vertices.contains(&lv) {
								return r.material;
							}
						}
						panic!("meshlet contains vertex but no material range contains it");
					}
				}
				Uuid::default()
			}),
			vertices: vertices.into_iter().map(|x| mesh.vertices[x as usize]).collect(),
			indices,
		}
	}

	fn generate_groups(&self, range: Range<usize>, mesh: &Mesh) -> Vec<Vec<usize>> {
		let s = span!(Level::TRACE, "grouping meshlets");
		let _e = s.enter();

		let connections = self.find_connections(range.clone(), mesh);

		let mut xadj = Vec::with_capacity(range.len() + 1);
		let mut adj = Vec::new();
		let mut weights = Vec::new();

		for mid in range.clone() {
			xadj.push(adj.len() as i32);
			for &(connected, count) in connections[mid - range.start].iter() {
				adj.push((connected - range.start) as i32);
				weights.push(count as i32);
			}
		}
		xadj.push(adj.len() as i32);

		let mut group_of = vec![0; range.len()];
		let group_count = range.len().div_ceil(4);
		Graph::new(1, group_count as _, &xadj, &adj)
			.unwrap()
			.set_adjwgt(&weights)
			.part_kway(&mut group_of)
			.unwrap();

		let mut groups = vec![Vec::new(); group_count];
		for (i, group) in group_of.into_iter().enumerate() {
			groups[group as usize].push(i + range.start);
		}
		groups
	}

	fn find_connections(&self, range: Range<usize>, mesh: &Mesh) -> Vec<Vec<(usize, usize)>> {
		let mut shared_edges = FxHashMap::default();

		for mid in range.clone() {
			let m = &mesh.meshlets[mid];
			for t in 0..m.tri_count {
				for i in 0..3 {
					let v0 = mesh.indices[m.index_offset as usize + t as usize * 3 + i] as u32 + m.vertex_offset;
					let v1 =
						mesh.indices[m.index_offset as usize + t as usize * 3 + (i + 1) % 3] as u32 + m.vertex_offset;
					let edge = (v0.min(v1), v0.max(v1));
					let out = shared_edges.entry(edge).or_insert(Vec::new());
					if out.last() != Some(&mid) {
						out.push(mid);
					}
				}
			}
		}

		let mut shared_count = BTreeMap::new();
		for (_, mids) in shared_edges {
			for i in 0..mids.len() {
				for j in i..mids.len() {
					let i = mids[i];
					let j = mids[j];
					*shared_count.entry((i.min(j), i.max(j))).or_default() += 1;
				}
			}
		}

		let mut connections = vec![Vec::new(); range.len()];
		for ((m1, m2), count) in shared_count {
			connections[m1 - range.start].push((m2, count));
			connections[m2 - range.start].push((m1, count));
		}

		connections
	}

	fn conv_to_mesh(&self, mesh: gltf::Mesh, materials: &[Uuid]) -> ImportResult<FullMesh> {
		let s = span!(Level::TRACE, "convert from gltf");
		let _e = s.enter();

		let mut out = FullMesh {
			vertices: Vec::with_capacity(
				mesh.primitives()
					.flat_map(|x| x.get(&gltf::Semantic::Positions).map(|x| x.count()))
					.sum(),
			),
			indices: Vec::with_capacity(mesh.primitives().flat_map(|x| x.indices().map(|x| x.count())).sum()),
			material_ranges: Vec::with_capacity(mesh.primitives().len()),
		};
		for prim in mesh.primitives() {
			let positions = prim.get(&gltf::Semantic::Positions).ok_or(ImportError::InvalidGltf)?;
			let (positions, ty, comp) = self.accessor(positions)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(ImportError::InvalidGltf);
			}
			let positions = positions.map(|p| *from_bytes::<Vec3<f32>>(p));

			let normals = prim.get(&gltf::Semantic::Normals).ok_or(ImportError::InvalidGltf)?;
			let (normals, ty, comp) = self.accessor(normals)?;
			if comp != Dimensions::Vec3 || ty != DataType::F32 {
				return Err(ImportError::InvalidGltf);
			}
			let normals = normals.map(|n| *from_bytes::<Vec3<f32>>(n));

			let tangents = prim.get(&gltf::Semantic::Tangents).ok_or(ImportError::InvalidGltf)?;
			let (tangents, ty, comp) = self.accessor(tangents)?;
			if comp != Dimensions::Vec4 || ty != DataType::F32 {
				return Err(ImportError::InvalidGltf);
			}
			let tangents = tangents.map(|n| *from_bytes::<Vec4<f32>>(n));

			let uv = prim.get(&gltf::Semantic::TexCoords(0));
			let mut uv = uv
				.map(|uv| {
					let (uv, ty, comp) = self.accessor(uv)?;
					if comp != Dimensions::Vec2 {
						return Err(ImportError::InvalidGltf);
					}

					if !matches!(ty, DataType::F32 | DataType::U8 | DataType::U16) {
						return Err(ImportError::InvalidGltf);
					}
					Ok(uv.map(move |uv| match ty {
						DataType::F32 => *from_bytes(uv),
						DataType::U8 => from_bytes::<Vec2<u8>>(uv).map(|u| u as f32 / 255.0),
						DataType::U16 => from_bytes::<Vec2<u16>>(uv).map(|u| u as f32 / 65535.0),
						_ => panic!("yikes"),
					}))
				})
				.transpose()?;

			let indices = prim.indices().ok_or(ImportError::InvalidGltf)?;
			let (indices, ty, comp) = self.accessor(indices)?;
			if comp != Dimensions::Scalar {
				return Err(ImportError::InvalidGltf);
			}
			let offset = out.vertices.len() as u32;
			match ty {
				DataType::U8 => out.indices.extend(indices.flatten().map(|&i| i as u32 + offset)),
				DataType::U16 => out
					.indices
					.extend(indices.map(|i| *from_bytes::<u16>(i) as u32 + offset)),
				DataType::U32 => out.indices.extend(indices.map(|i| *from_bytes::<u32>(i) + offset)),
				_ => return Err(ImportError::InvalidGltf),
			}

			out.vertices.extend(
				positions
					.zip(normals)
					.zip(tangents)
					.zip(std::iter::from_fn(move || {
						if let Some(ref mut uv) = uv {
							uv.next()
						} else {
							Some(Vec2::new(0.0, 0.0))
						}
					}))
					.map(|(((position, normal), tangent), uv)| Vertex {
						position,
						normal,
						tangent,
						uv,
					}),
			);

			out.material_ranges.push(FullMaterialRange {
				material: materials[prim.material().index().expect("TODO: work with default material")],
				vertices: offset..(out.vertices.len() as _),
			});
		}

		Ok(out)
	}
}

#[derive(Debug, PartialEq)]
struct FullMaterialRange {
	material: Uuid,
	vertices: Range<u32>,
}

struct FullMesh {
	vertices: Vec<Vertex>,
	indices: Vec<u32>,
	material_ranges: Vec<FullMaterialRange>,
}

impl FullMesh {
	fn get_material_of_vertex(&self, vertex: u32) -> Uuid {
		let mut start = 0;
		let mut end = self.material_ranges.len();
		loop {
			let mid = start + (end - start) / 2;
			let r = &self.material_ranges[mid as usize];
			if r.vertices.contains(&vertex) {
				return r.material;
			} else if vertex < r.vertices.start {
				end = mid;
			} else {
				start = mid;
			}
		}
	}
}

fn sort_by_material(
	vertices: &mut [u32], indices: &mut [u32], mut mat_src: impl FnMut(u32) -> Uuid,
) -> Vec<FullMaterialRange> {
	let mut ranges = Vec::new();
	let mut last_mat = mat_src(0);
	ranges.push(FullMaterialRange {
		material: last_mat,
		vertices: 0..0,
	});
	let mut i = 1;
	while i < vertices.len() {
		let mat = mat_src(vertices[i]);
		if mat != last_mat {
			let mut j = i + 1;
			while j < vertices.len() {
				if mat_src(vertices[j]) == last_mat {
					vertices.swap(i, j);
					for x in &mut *indices {
						let i = i as u32;
						let j = j as u32;
						if *x == i {
							*x = j;
						} else if *x == j {
							*x = i;
						}
					}
					break;
				}
				j += 1
			}
		}

		let next_mat = mat_src(vertices[i]);
		if next_mat != last_mat {
			ranges.last_mut().unwrap().vertices.end = i as u32;
			ranges.push(FullMaterialRange {
				material: next_mat,
				vertices: (i as u32)..0,
			});
		}
		last_mat = next_mat;
		i += 1;
	}
	ranges.last_mut().unwrap().vertices.end = i as u32;

	ranges
}

#[test]
fn mat_sort() {
	let m1 = Uuid::from_u64_pair(0, 1);
	let m2 = Uuid::from_u64_pair(0, 2);
	let mesh = FullMesh {
		vertices: vec![Default::default(); 10],
		indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		material_ranges: vec![
			FullMaterialRange {
				material: m1,
				vertices: 0..5,
			},
			FullMaterialRange {
				material: m2,
				vertices: 5..10,
			},
		],
	};
	let mut vertices = vec![0, 5, 1, 6, 2, 7, 3, 8, 4, 9];
	let mut indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
	let ranges = sort_by_material(&mut vertices, &mut indices, |x| mesh.get_material_of_vertex(x));
	assert_eq!(vertices.as_slice(), &[0, 1, 2, 3, 4, 7, 6, 8, 5, 9]);
	assert_eq!(indices.as_slice(), &[0, 8, 1, 6, 2, 5, 3, 7, 4, 9]);
	assert_eq!(
		ranges.as_slice(),
		&[
			FullMaterialRange {
				material: m1,
				vertices: 0..5
			},
			FullMaterialRange {
				material: m2,
				vertices: 5..10
			}
		]
	);
}
