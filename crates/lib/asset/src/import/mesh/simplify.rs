use std::{
	cmp::Ordering,
	collections::{hash_map::Entry, BinaryHeap},
	ops::Add,
};

use rustc_hash::{FxHashMap, FxHashSet};
use vek::{Mat4, Vec2, Vec3, Vec4};

use crate::{import::mesh::FullMesh, mesh::Vertex};

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
struct Edge {
	v0: u32,
	v1: u32,
	dirty: bool,
	collapsed: bool,
}

impl Edge {
	fn new(v0: u32, v1: u32) -> Self {
		Edge {
			v0: v0.min(v1),
			v1: v0.max(v1),
			dirty: false,
			collapsed: false,
		}
	}

	fn other(&self, v: u32) -> u32 {
		if v == self.v0 {
			self.v1
		} else {
			self.v0
		}
	}
}

struct EdgeVertex {
	vertex: Vertex,
	edges: FxHashSet<u32>,
	tris: FxHashSet<u32>,
	quadrics: Mat4<f32>,
}

struct EdgeTri {
	vertices: [u32; 3],
	collapsed: bool,
}

struct EdgeMesh {
	vertices: Vec<EdgeVertex>,
	edges: Vec<Edge>,
	tris: Vec<EdgeTri>,
	boundary: FxHashSet<u32>,
}

fn remap_vertices(vertices: &[Vertex], indices: impl IntoIterator<Item = u32>) -> (Vec<Vertex>, Vec<u32>) {
	let mut map = FxHashMap::default();
	let mut v = Vec::new();
	let i = indices
		.into_iter()
		.map(|i| {
			*map.entry(i).or_insert_with(|| {
				let n = v.len();
				v.push(vertices[i as usize]);
				n as u32
			})
		})
		.collect();
	(v, i)
}

fn edges_of(tri: [u32; 3]) -> [Edge; 3] {
	[
		Edge::new(tri[0], tri[1]),
		Edge::new(tri[0], tri[2]),
		Edge::new(tri[1], tri[2]),
	]
}

impl EdgeMesh {
	fn new(vertices: &[Vertex], indices: &[u32]) -> Self {
		let mut edge_map = FxHashMap::default();
		let mut boundary = FxHashSet::default();
		let mut edges = Vec::with_capacity(indices.len() * 2);

		let (v, i) = remap_vertices(vertices, indices.iter().copied());
		let mut vertices: Vec<_> = v
			.into_iter()
			.map(|vertex| EdgeVertex {
				vertex,
				edges: FxHashSet::default(),
				tris: FxHashSet::default(),
				quadrics: Mat4::zero(),
			})
			.collect();

		let tris = i
			.chunks(3)
			.enumerate()
			.map(|(i, tri)| {
				let ivertices = [tri[0], tri[1], tri[2]];
				let es = edges_of(ivertices);
				for e in es {
					match edge_map.entry(e) {
						Entry::Occupied(o) => {
							boundary.remove(o.get());
						},
						Entry::Vacant(v) => {
							let i = edges.len();
							edges.push(e);
							let e = i as u32;
							boundary.insert(e);
							v.insert(e);
						},
					}
				}
				for (v, e) in ivertices.into_iter().map(|x| (x, es)) {
					let vtex = &mut vertices[v as usize];
					vtex.tris.insert(i as _);
					vtex.edges.extend(
						e.into_iter()
							.filter_map(|x| (x.v0 == v || x.v1 == v).then_some(edge_map[&x])),
					);
				}

				EdgeTri {
					vertices: ivertices,
					collapsed: false,
				}
			})
			.collect();

		let mut this = Self {
			vertices,
			tris,
			edges,
			boundary,
		};
		this.calc_quadrics();
		this
	}

	fn finish(self) -> FullMesh {
		let v: Vec<_> = self.vertices.into_iter().map(|x| x.vertex).collect();
		let (vertices, indices) = remap_vertices(
			&v,
			self.tris.into_iter().filter(|x| !x.collapsed).flat_map(|x| x.vertices),
		);
		FullMesh { vertices, indices }
	}

	fn calc_quadrics_of(&mut self, vertex: u32) {
		let q = self
			.tris_of(vertex)
			.map(|t| mul_transpose(self.plane_of(t)))
			.reduce(mat_sum)
			.unwrap_or(Mat4::zero());
		self.vertices[vertex as usize].quadrics = q;
	}

	fn calc_quadrics(&mut self) {
		for v in 0..self.vertices.len() {
			self.calc_quadrics_of(v as _);
		}
	}

	fn edge_cost(&self, edge: u32) -> (Vec3<f32>, f32) {
		let e = self.edges[edge as usize];
		let v0 = &self.vertices[e.v0 as usize];
		let v1 = &self.vertices[e.v1 as usize];
		let q0 = v0.quadrics;
		let q1 = v1.quadrics;
		let q = mat_sum(q0, q1);

		let mut minimization = q.into_row_arrays();
		minimization[3][..3].fill(0.0);
		minimization[3][3] = 1.0;
		let minimization = Mat4::from_row_arrays(minimization);

		let v = if minimization.determinant() > 0.001 {
			minimization.inverted() * Vec3::zero().with_w(1.0)
		} else {
			((v0.vertex.position + v1.vertex.position) * 0.5).with_w(1.0)
		};

		(v.xyz(), calc_error(v, q).abs())
	}

	fn vertices_of(&self, edge: u32) -> (u32, u32) {
		let e = self.edges[edge as usize];
		(e.v0, e.v1)
	}

	fn tris_of(&self, vertex: u32) -> impl Iterator<Item = u32> + '_ {
		self.vertices[vertex as usize].tris.iter().copied()
	}

	fn edges_of(&self, vertex: u32) -> impl Iterator<Item = u32> + '_ {
		self.vertices[vertex as usize].edges.iter().copied()
	}

	fn neighbouring_vertices(&self, vertex: u32) -> impl Iterator<Item = u32> + '_ {
		self.edges_of(vertex).map(move |e| self.edges[e as usize].other(vertex))
	}

	fn plane_of(&self, tri: u32) -> Vec4<f32> {
		let t = &self.tris[tri as usize];
		let v0 = self.vertices[t.vertices[0] as usize].vertex.position;
		let v1 = self.vertices[t.vertices[1] as usize].vertex.position;
		let v2 = self.vertices[t.vertices[2] as usize].vertex.position;

		let d1 = v1 - v0;
		let d2 = v2 - v0;
		let n = d1.cross(d2).normalized();
		let d = -n.dot(v0);
		n.with_w(d)
	}

	fn is_collapse_safe(&self, edge: u32) -> bool {
		if self.boundary.contains(&edge) {
			return false;
		}
		let (s, e) = self.vertices_of(edge);
		for v in [s, e] {
			for e in self.edges_of(v) {
				if self.boundary.contains(&e) {
					return false;
				}
			}
		}

		let s_neighbours: FxHashSet<_> = self.neighbouring_vertices(s).collect();
		let common_neighbours = self
			.neighbouring_vertices(e)
			.filter(|x| s_neighbours.contains(&x))
			.count();
		common_neighbours == 2
	}

	fn barycentrics_of(&self, pos: Vec3<f32>, tri: u32) -> Vec3<f32> {
		let t = &self.tris[tri as usize];
		let a = self.vertices[t.vertices[0] as usize].vertex.position;
		let b = self.vertices[t.vertices[1] as usize].vertex.position;
		let c = self.vertices[t.vertices[2] as usize].vertex.position;

		let v0 = b - a;
		let v1 = c - a;
		let v2 = pos - a;
		let d00 = v0.dot(v0);
		let d01 = v0.dot(v1);
		let d11 = v1.dot(v1);
		let d20 = v2.dot(v0);
		let d21 = v2.dot(v1);
		let denom = d00 * d11 - d01 * d01;
		let v = (d11 * d20 - d01 * d21) / denom;
		let w = (d00 * d21 - d01 * d20) / denom;
		let u = 1.0 - v - w;

		Vec3::new(u, v, w)
	}

	fn get_vertex_of_collapse(&self, edge: u32, break_pos: Vec3<f32>) -> Vertex {
		let _ = &self.edges[edge as usize];
		// TODO: find out why no work
		// for v in [e.v0, e.v1] {
		// 	for t in self.tris_of(v).filter(|&x| {
		// 		let t = &self.tris[x as usize];
		// 		t.vertices.contains(&e.v0) && t.vertices.contains(&e.v1)
		// 	}) {
		// 		let bary = self.barycentrics_of(break_pos, t);
		// 		if bary.into_iter().all(|x| x >= -0.1) {
		// 			let t = &self.tris[t as usize];
		// 			let v0 = self.vertices[t.vertices[0] as usize].vertex;
		// 			let v1 = self.vertices[t.vertices[1] as usize].vertex;
		// 			let v2 = self.vertices[t.vertices[2] as usize].vertex;
		//
		// 			let normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
		// 			let uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;
		// 			return Vertex {
		// 				position: break_pos,
		// 				normal,
		// 				uv,
		// 			};
		// 		}
		// 	}
		// }
		// panic!("couldn't find triangle for collapsed vertex {break_pos:?}");
		Vertex {
			position: break_pos,
			normal: Vec3::zero(),
			uv: Vec2::zero(),
		}
	}

	fn collapse(&mut self, edge: u32, break_pos: Vec3<f32>, q: &mut BinaryHeap<Candidate>) {
		let new_vertex = self.get_vertex_of_collapse(edge, break_pos);
		let id = self.vertices.len() as u32;
		self.vertices.push(EdgeVertex {
			vertex: new_vertex,
			edges: FxHashSet::default(),
			tris: FxHashSet::default(),
			quadrics: Mat4::zero(),
		});

		let e = &mut self.edges[edge as usize];
		e.collapsed = true;
		let v0 = e.v0;
		let v1 = e.v1;

		let mut edges = FxHashSet::default();
		let mut tris = FxHashSet::default();
		let tis: Vec<_> = [v0, v1].into_iter().flat_map(|v| self.tris_of(v)).collect();
		let es: Vec<_> = [v0, v1].into_iter().flat_map(|v| self.edges_of(v)).collect();

		for ti in tis {
			let n = self.plane_of(ti).xyz();

			let t = &mut self.tris[ti as usize];
			let mut swapped = false;
			for v in t.vertices.iter_mut() {
				if *v == v0 || *v == v1 {
					if swapped {
						t.collapsed = true;
						break;
					}
					*v = id;
					swapped = true;
				}
			}

			if !t.collapsed {
				let n2 = self.plane_of(ti).xyz();
				if n.dot(n2) < 0.0 {
					let t = &mut self.tris[ti as usize];
					let [v0, v1, v2] = t.vertices;
					t.vertices = [v1, v0, v2];
				}
				tris.insert(ti);
			}
		}

		for e in es {
			if e != edge {
				edges.insert(e);

				let ed = &mut self.edges[e as usize];
				ed.dirty = true;
				if self.is_collapse_safe(e) {
					let (break_pos, cost) = self.edge_cost(e);
					q.push(Candidate { cost, edge, break_pos });
				}
			}
		}

		self.vertices[id as usize].edges = edges;
		self.vertices[id as usize].tris = tris;
		self.calc_quadrics_of(id);
	}
}

fn mul_transpose(v: Vec4<f32>) -> Mat4<f32> { Mat4 { cols: v.map(|x| x * v) } }

fn mat_sum(a: Mat4<f32>, b: Mat4<f32>) -> Mat4<f32> { a.map2(b, Add::add) }

fn calc_error(v: Vec4<f32>, q: Mat4<f32>) -> f32 {
	let vt_q = q.cols.map(|x| x.dot(v));
	vt_q.dot(v)
}

struct Candidate {
	cost: f32,
	edge: u32,
	break_pos: Vec3<f32>,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
	fn eq(&self, other: &Self) -> bool { self.cost == other.cost && self.edge == other.edge }
}

impl Ord for Candidate {
	fn cmp(&self, other: &Self) -> Ordering {
		other
			.cost
			.partial_cmp(&self.cost)
			.unwrap_or(Ordering::Less)
			.then_with(|| self.edge.cmp(&other.edge))
	}
}

impl PartialOrd for Candidate {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

pub fn simplify_mesh(vertices: &[Vertex], indices: &[u32]) -> Option<(FullMesh, f32)> {
	let mut mesh = EdgeMesh::new(vertices, indices);

	let mut q = BinaryHeap::new();
	for e in 0..mesh.edges.len() {
		let e = e as u32;
		if mesh.is_collapse_safe(e) {
			let (break_pos, cost) = mesh.edge_cost(e);
			q.push(Candidate {
				cost,
				edge: e,
				break_pos,
			});
		}
	}

	let mut remaining = mesh.tris.len();
	let target = mesh.tris.len() / 2;
	let mut total_cost = 0.0f32;
	while remaining > target && !q.is_empty() {
		let Candidate { cost, edge, break_pos } = q.pop().unwrap();
		let e = &mesh.edges[edge as usize];

		if e.collapsed {
			continue;
		} else if e.dirty {
			mesh.edges[edge as usize].dirty = false;
			if mesh.is_collapse_safe(edge) {
				let (break_pos, cost) = mesh.edge_cost(edge);
				q.push(Candidate { cost, edge, break_pos });
			}
			continue;
		} else if !mesh.is_collapse_safe(edge) {
			continue;
		}

		mesh.collapse(edge, break_pos, &mut q);
		remaining -= 2;
		total_cost = total_cost.max(cost);
	}

	let ratio = target as f32 / remaining as f32;
	// TODO: should this be divided by 2?
	(ratio > 0.8).then(|| (mesh.finish(), total_cost.sqrt() * 0.5))
}
