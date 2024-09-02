use std::{collections::BTreeMap, ops::Range};

use bytemuck::from_bytes;
use gltf::accessor::{DataType, Dimensions};
use meshopt::VertexDataAdapter;
use metis::Graph;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use tracing::{debug_span, field, info_span, trace_span};
use uuid::Uuid;
use vek::{Aabb, Vec2, Vec3};

use crate::{
	import::{mesh::bvh::aabb_to_sphere, ImportError, ImportResult, Importer},
	mesh::{BvhNode, Mesh, Meshlet, Vertex},
};

mod bvh;
mod simplify;

impl Meshlet {
	fn vertices(&self) -> Range<usize> {
		(self.vert_offset as usize)..(self.vert_offset as usize + self.vert_count as usize)
	}

	fn tris(&self) -> Range<usize> {
		(self.index_offset as usize)..(self.index_offset as usize + self.tri_count as usize * 3)
	}
}

#[derive(Clone)]
struct MeshletGroup {
	aabb: Aabb<f32>,
	lod_bounds: Aabb<f32>,
	parent_error: f32,
	meshlets: Range<u32>,
}

struct Meshlets {
	vertex_remap: Vec<u32>,
	tris: Vec<u8>,
	groups: Vec<MeshletGroup>,
	meshlets: Vec<Meshlet>,
	lod_bounds: Vec<Aabb<f32>>,
}

impl Meshlets {
	fn add(&mut self, other: Meshlets) {
		let vertex_offset = self.vertex_remap.len() as u32;
		let tri_offset = self.tris.len() as u32;
		let meshlet_offset = self.meshlets.len() as u32;
		self.meshlets.extend(other.meshlets.into_iter().map(|mut m| {
			m.vert_offset += vertex_offset;
			m.index_offset += tri_offset;
			m
		}));
		self.lod_bounds.extend(other.lod_bounds);
		self.vertex_remap.extend(other.vertex_remap);
		self.tris.extend(other.tris);
		self.groups.extend(other.groups.into_iter().map(|mut g| {
			g.meshlets.start += meshlet_offset;
			g.meshlets.end += meshlet_offset;
			g
		}));
	}
}

impl Importer<'_> {
	pub fn mesh(&self, name: &str, mesh: gltf::Mesh, materials: &[Uuid]) -> ImportResult<Mesh> {
		let s = info_span!("importing mesh", name = name);
		let _e = s.enter();

		let mesh = self.conv_to_mesh(mesh, materials)?;
		let mut meshlets = self.generate_meshlets(&mesh.vertices, &mesh.indices, None);

		let mut bvh = bvh::BvhBuilder::default();
		let mut simplify = 0..meshlets.meshlets.len();
		let mut lod = 1;
		while simplify.len() > 1 {
			let s = debug_span!(
				"generating lod",
				lod,
				meshlets = simplify.len(),
				groups = field::Empty,
				min_error = field::Empty,
				avg_error = field::Empty,
				max_error = field::Empty,
			);
			let _e = s.enter();

			let next_start = meshlets.meshlets.len();

			let first = self.generate_groups(simplify.clone(), &mut meshlets);

			// self.dump_groups_to_obj(name, lod, &mesh.vertices, &meshlets, first);

			s.record("groups", meshlets.groups.len() - first);
			let par: Vec<_> = meshlets.groups[first..]
				.into_par_iter()
				.enumerate()
				.filter(|(_, x)| x.meshlets.len() > 1)
				.filter_map(|(i, group)| {
					let (indices, parent_error) = self.simplify_group(&mesh.vertices, &meshlets, &group)?;
					let n_meshlets =
						self.generate_meshlets(&mesh.vertices, &indices, Some((group.lod_bounds, parent_error)));
					Some((i, parent_error, n_meshlets))
				})
				.collect();
			let mut min_error = f32::MAX;
			let mut avg_error = 0.0f32;
			let mut max_error = 0.0f32;
			let count = par.len();
			for (group, parent_error, n_meshlets) in par {
				meshlets.add(n_meshlets);
				meshlets.groups[group + first].parent_error = parent_error;

				min_error = min_error.min(parent_error);
				avg_error += parent_error;
				max_error = max_error.max(parent_error);
			}
			if count > 0 {
				s.record("min_error", min_error);
				s.record("avg_error", avg_error / count as f32);
				s.record("max_error", max_error);
			}

			bvh.add_lod(first as _, &meshlets.groups[first..]);

			simplify = next_start..meshlets.meshlets.len();
			lod += 1;
		}

		let (bvh, depth) = bvh.build(&meshlets.groups);
		Ok(self.convert_meshlets(mesh.vertices, meshlets, bvh, depth))
	}

	fn generate_meshlets(&self, vertices: &[Vertex], indices: &[u32], error: Option<(Aabb<f32>, f32)>) -> Meshlets {
		let s = trace_span!("building meshlets");
		let _e = s.enter();

		let adapter = VertexDataAdapter::new(bytemuck::cast_slice(vertices), std::mem::size_of::<Vertex>(), 0).unwrap();
		let ms = meshopt::build_meshlets(indices, &adapter, 128, 124, 0.0);
		let (meshlets, lod_bounds) = ms
			.meshlets
			.iter()
			.map(|m| {
				let m_vertices = &ms.vertices[m.vertex_offset as usize..(m.vertex_offset + m.vertex_count) as usize];
				let m_indices =
					&ms.triangles[m.triangle_offset as usize..(m.triangle_offset + m.triangle_count * 3) as usize];
				let aabb = bvh::calc_aabb(m_vertices.iter().map(|&x| &vertices[x as usize]));
				let (lod_bounds, error) = error.unwrap_or((aabb, 0.0));
				let mut max_edge_length = 0.0f32;
				for t in m_indices.chunks(3) {
					for (v1, v2) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
						max_edge_length = max_edge_length.max(
							(vertices[m_vertices[v1 as usize] as usize].position
								- vertices[m_vertices[v2 as usize] as usize].position)
								.magnitude_squared(),
						);
					}
				}
				max_edge_length = max_edge_length.sqrt();

				(
					Meshlet {
						vert_offset: m.vertex_offset,
						vert_count: m.vertex_count as _,
						index_offset: m.triangle_offset,
						tri_count: m.triangle_count as _,
						aabb,
						lod_bounds: aabb_to_sphere(lod_bounds),
						error,
						max_edge_length,
					},
					lod_bounds,
				)
			})
			.unzip();

		Meshlets {
			vertex_remap: ms.vertices,
			tris: ms.triangles,
			groups: Vec::new(),
			meshlets,
			lod_bounds,
		}
	}

	fn generate_groups(&self, range: Range<usize>, meshlets: &mut Meshlets) -> usize {
		let s = trace_span!("grouping meshlets");
		let _e = s.enter();

		// TODO: locality links
		let connections = self.find_connections(range.clone(), meshlets);

		let mut xadj = Vec::with_capacity(range.len() + 1);
		let mut adj = Vec::new();
		let mut weights = Vec::new();

		for conn in connections {
			xadj.push(adj.len() as i32);
			for (connected, count) in conn {
				adj.push(connected as i32);
				weights.push(count as i32);
			}
		}
		xadj.push(adj.len() as i32);

		let mut group_of = vec![0; range.len()];
		let group_count = range.len().div_ceil(8);
		Graph::new(1, group_count as _, &xadj, &adj)
			.unwrap()
			.set_adjwgt(&weights)
			.part_kway(&mut group_of)
			.unwrap();

		let first = meshlets.groups.len();

		let mut meshlet_reorder: Vec<_> = range.clone().collect();
		meshlet_reorder.sort_unstable_by_key(|&x| group_of[x - range.start]);

		let mut out = vec![Meshlet::default(); meshlet_reorder.len()];
		let mut last_group = 0;
		let mut group = MeshletGroup {
			aabb: bvh::aabb_default(),
			lod_bounds: bvh::aabb_default(),
			parent_error: f32::MAX,
			meshlets: range.start as u32..0,
		};
		for (i, p) in meshlet_reorder.into_iter().enumerate() {
			let next = group_of[p - range.start];
			if last_group != next {
				let end = (i + range.start) as u32;
				group.meshlets.end = end;
				meshlets.groups.push(group.clone());

				last_group = next;
				group.aabb = bvh::aabb_default();
				group.lod_bounds = bvh::aabb_default();
				group.meshlets.start = end;
			}

			let m = meshlets.meshlets[p];
			group.aabb = group.aabb.union(m.aabb);
			group.lod_bounds = group.lod_bounds.union(meshlets.lod_bounds[p]);
			out[i] = m;
		}
		group.meshlets.end = (out.len() + range.start) as u32;
		meshlets.groups.push(group);

		meshlets.meshlets[range.start..].copy_from_slice(&out);

		first
	}

	fn simplify_group(
		&self, vertices: &[Vertex], meshlets: &Meshlets, group: &MeshletGroup,
	) -> Option<(Vec<u32>, f32)> {
		let s = trace_span!("simplifying group");
		let _e = s.enter();

		let ms = &meshlets.meshlets[(group.meshlets.start as usize)..(group.meshlets.end as usize)];
		let indices: Vec<_> = ms
			.iter()
			.flat_map(|m| {
				let verts = &meshlets.vertex_remap[m.vertices()];
				meshlets.tris[m.tris()].iter().map(move |&x| verts[x as usize])
			})
			.collect();

		let adapter = VertexDataAdapter::new(bytemuck::cast_slice(vertices), std::mem::size_of::<Vertex>(), 0).unwrap();

		let mut error = 0.0;
		let simplified = meshopt::simplify(
			&indices,
			&adapter,
			indices.len() / 2,
			group.aabb.size().reduce_partial_max(),
			meshopt::SimplifyOptions::LockBorder
				| meshopt::SimplifyOptions::Sparse
				| meshopt::SimplifyOptions::ErrorAbsolute,
			Some(&mut error),
		);
		error *= 0.5;

		for m in ms {
			error = error.max(m.error);
		}

		if (simplified.len() as f32 / indices.len() as f32) < 0.65 {
			Some((simplified, error))
		} else {
			None
		}
	}

	fn find_connections(&self, range: Range<usize>, meshlets: &Meshlets) -> Vec<Vec<(usize, usize)>> {
		let s = trace_span!("generating meshlet graph");
		let _e = s.enter();

		let mut shared_edges = FxHashMap::default();
		for mid in range.clone() {
			let m = &meshlets.meshlets[mid];
			let verts = &meshlets.vertex_remap[m.vertices()];
			for i in meshlets.tris[m.tris()].chunks(3) {
				for j in 0..3 {
					let i0 = i[j] as usize;
					let i1 = i[(j + 1) % 3] as usize;
					let v0 = verts[i0];
					let v1 = verts[i1];
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
				for j in (i + 1)..mids.len() {
					let i = mids[i];
					let j = mids[j];
					*shared_count.entry((i.min(j), i.max(j))).or_default() += 1;
				}
			}
		}

		let mut connections = vec![Vec::new(); range.len()];
		for ((mut m1, mut m2), count) in shared_count {
			m1 -= range.start;
			m2 -= range.start;
			connections[m1].push((m2, count));
			connections[m2].push((m1, count));
		}
		connections
	}

	fn convert_meshlets(&self, vertices: Vec<Vertex>, meshlets: Meshlets, bvh: Vec<BvhNode>, bvh_depth: u32) -> Mesh {
		let mut outv = Vec::with_capacity(vertices.len());
		let mut outi = Vec::with_capacity(meshlets.meshlets.len() * 124 * 3);
		let meshlets = meshlets
			.meshlets
			.into_iter()
			.map(|m| {
				let indices: Vec<_> = meshlets.tris[m.tris()].iter().map(|&x| x as u32).collect();
				let vertices = meshlets.vertex_remap[m.vertices()]
					.iter()
					.map(|&x| vertices[x as usize]);
				let index_offset = outi.len() as u32;
				let vert_offset = outv.len() as u32;
				outv.extend(vertices);
				outi.extend(indices.into_iter().map(|x| x as u8));
				Meshlet {
					vert_offset,
					index_offset,
					..m
				}
			})
			.collect();
		let aabb = bvh::calc_aabb(&outv);

		Mesh {
			vertices: outv,
			indices: outi,
			meshlets,
			bvh,
			bvh_depth,
			aabb,
		}
	}

	fn conv_to_mesh(&self, mesh: gltf::Mesh, _: &[Uuid]) -> ImportResult<FullMesh> {
		let s = trace_span!("convert from gltf");
		let _e = s.enter();

		let mut out = FullMesh {
			vertices: Vec::with_capacity(
				mesh.primitives()
					.flat_map(|x| x.get(&gltf::Semantic::Positions).map(|x| x.count()))
					.sum(),
			),
			indices: Vec::with_capacity(mesh.primitives().flat_map(|x| x.indices().map(|x| x.count())).sum()),
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
					.zip(std::iter::from_fn(move || {
						if let Some(ref mut uv) = uv {
							uv.next()
						} else {
							Some(Vec2::new(0.0, 0.0))
						}
					}))
					.map(|((position, normal), uv)| Vertex { position, normal, uv }),
			);
		}

		Ok(out)
	}

	#[cfg(feature = "disabled")]
	fn dump_groups_to_obj(&self, name: &str, lod: usize, vertices: &[Vertex], meshlets: &Meshlets, first: usize) {
		let mut o = String::new();
		for v in vertices.iter() {
			o.push_str(&format!("v {} {} {}\n", v.position.x, v.position.y, v.position.z));
		}
		for (i, g) in meshlets.groups[first..].iter().enumerate() {
			o.push_str(&format!("o group {}\n", i));
			for m in meshlets.meshlets[(g.meshlets.start as usize)..(g.meshlets.end as usize)].iter() {
				let verts = &meshlets.vertex_remap[m.vertices()];
				for t in meshlets.tris[m.tris()].chunks(3) {
					o.push_str(&format!(
						"f {} {} {}\n",
						verts[t[0] as usize] + 1,
						verts[t[1] as usize] + 1,
						verts[t[2] as usize] + 1,
					));
				}
			}
		}
		std::fs::write(format!("{name}_grouped_{lod}.obj"), o).unwrap();
	}
}

struct FullMesh {
	vertices: Vec<Vertex>,
	indices: Vec<u32>,
}
