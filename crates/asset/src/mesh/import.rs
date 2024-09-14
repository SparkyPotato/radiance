use std::{cell::RefCell, collections::BTreeMap, ops::Range};

use meshopt::VertexDataAdapter;
use metis::Graph;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};
use thread_local::ThreadLocal;
use tracing::{debug_span, field, info_span, trace_span};
use vek::{Aabb, Sphere, Vec3};

use crate::mesh::data::{BvhNode, Mesh, Meshlet, Vertex};

impl Meshlet {
	fn vertices(&self) -> Range<usize> {
		(self.vert_offset as usize)..(self.vert_offset as usize + self.vert_count as usize)
	}

	fn tris(&self) -> Range<usize> {
		(self.index_offset as usize)..(self.index_offset as usize + self.tri_count as usize * 3)
	}
}

pub struct FullMesh {
	pub vertices: Vec<Vertex>,
	pub indices: Vec<u32>,
}

#[derive(Clone)]
struct MeshletGroup {
	aabb: Aabb<f32>,
	lod_bounds: Sphere<f32, f32>,
	parent_error: f32,
	meshlets: Range<u32>,
}

struct Meshlets {
	vertex_remap: Vec<u32>,
	tris: Vec<u8>,
	groups: Vec<MeshletGroup>,
	meshlets: Vec<Meshlet>,
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
		self.vertex_remap.extend(other.vertex_remap);
		self.tris.extend(other.tris);
		self.groups.extend(other.groups.into_iter().map(|mut g| {
			g.meshlets.start += meshlet_offset;
			g.meshlets.end += meshlet_offset;
			g
		}));
	}
}

pub fn import(name: &str, mesh: FullMesh) -> Mesh {
	let s = info_span!("importing mesh", name = name);
	let _e = s.enter();

	let mut boundary = vec![false; mesh.vertices.len()];
	compute_boundary(&mesh.indices, &mut boundary);
	let mut meshlets = generate_meshlets(&mesh.vertices, &mesh.indices, None);

	let mut bvh = BvhBuilder::default();
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

		let first = generate_groups(simplify.clone(), &mut meshlets);

		// self.dump_groups_to_obj(name, lod, &mesh.vertices, &meshlets, first);

		s.record("groups", meshlets.groups.len() - first);
		let tls = ThreadLocal::new();
		let par: Vec<_> = meshlets.groups[first..]
			.into_par_iter()
			.enumerate()
			.filter(|(_, x)| x.meshlets.len() > 1)
			.filter_map(|(i, group)| {
				let tls = tls.get_or(|| RefCell::new(vec![false; mesh.vertices.len()]));
				let (indices, parent_error) = simplify_group(
					&mesh.vertices,
					&boundary,
					&meshlets,
					&group,
					tls.borrow_mut().as_mut_slice(),
				)?;
				let n_meshlets = generate_meshlets(&mesh.vertices, &indices, Some((group.lod_bounds, parent_error)));
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
	convert_meshlets(mesh.vertices, meshlets, bvh, depth)
}

fn generate_meshlets(vertices: &[Vertex], indices: &[u32], error: Option<(Sphere<f32, f32>, f32)>) -> Meshlets {
	let s = trace_span!("building meshlets");
	let _e = s.enter();

	let adapter = VertexDataAdapter::new(bytemuck::cast_slice(vertices), std::mem::size_of::<Vertex>(), 0).unwrap();
	let ms = meshopt::build_meshlets(indices, &adapter, 128, 124, 0.0);
	let meshlets = ms
		.meshlets
		.iter()
		.enumerate()
		.map(|(i, m)| {
			let m_vertices = &ms.vertices[m.vertex_offset as usize..(m.vertex_offset + m.vertex_count) as usize];
			let m_indices =
				&ms.triangles[m.triangle_offset as usize..(m.triangle_offset + m.triangle_count * 3) as usize];
			let aabb = calc_aabb(m_vertices.iter().map(|&x| &vertices[x as usize]));
			let bounds = meshopt::compute_meshlet_bounds(ms.get(i), &adapter);
			let (lod_bounds, error) = error.unwrap_or((
				Sphere {
					center: bounds.center.into(),
					radius: bounds.radius,
				},
				0.0,
			));
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

			Meshlet {
				vert_offset: m.vertex_offset,
				vert_count: m.vertex_count as _,
				index_offset: m.triangle_offset,
				tri_count: m.triangle_count as _,
				aabb,
				lod_bounds,
				error,
				max_edge_length,
			}
		})
		.collect();

	Meshlets {
		vertex_remap: ms.vertices,
		tris: ms.triangles,
		groups: Vec::new(),
		meshlets,
	}
}

fn generate_groups(range: Range<usize>, meshlets: &mut Meshlets) -> usize {
	let s = trace_span!("grouping meshlets");
	let _e = s.enter();

	// TODO: locality links
	let connections = find_connections(range.clone(), meshlets);

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
		aabb: aabb_default(),
		lod_bounds: Sphere::default(),
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
			group.aabb = aabb_default();
			group.lod_bounds = Sphere::default();
			group.meshlets.start = end;
		}

		let m = meshlets.meshlets[p];
		group.aabb = group.aabb.union(m.aabb);
		group.lod_bounds = merge_spheres(group.lod_bounds, m.lod_bounds);
		out[i] = m;
	}
	group.meshlets.end = (out.len() + range.start) as u32;
	meshlets.groups.push(group);

	meshlets.meshlets[range.start..].copy_from_slice(&out);

	first
}

fn simplify_group(
	vertices: &[Vertex], mesh_boundary: &[bool], meshlets: &Meshlets, group: &MeshletGroup, group_boundary: &mut [bool],
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

	compute_boundary(&indices, group_boundary);
	for (g, &m) in group_boundary.iter_mut().zip(mesh_boundary) {
		*g = *g && !m;
	}

	let adapter = VertexDataAdapter::new(bytemuck::cast_slice(vertices), std::mem::size_of::<Vertex>(), 0).unwrap();
	let mut error = 0.0;
	let simplified = meshopt::simplify_with_locks(
		&indices,
		&adapter,
		group_boundary,
		indices.len() / 2,
		f32::MAX,
		meshopt::SimplifyOptions::Sparse | meshopt::SimplifyOptions::ErrorAbsolute,
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

fn find_connections(range: Range<usize>, meshlets: &Meshlets) -> Vec<Vec<(usize, usize)>> {
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

fn convert_meshlets(vertices: Vec<Vertex>, meshlets: Meshlets, bvh: Vec<BvhNode>, bvh_depth: u32) -> Mesh {
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
	let aabb = calc_aabb(&outv);

	Mesh {
		vertices: outv,
		indices: outi,
		meshlets,
		bvh,
		bvh_depth,
		aabb,
	}
}

fn compute_boundary(indices: &[u32], out: &mut [bool]) {
	let mut edge_set = FxHashSet::default();
	let mut edge_edge_set = FxHashSet::default();
	for tri in indices.chunks(3) {
		for (v0, v1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])] {
			let edge = (v0.min(v1), v0.max(v1));
			if edge_set.insert(edge) {
				// Never seen this before, it's on the boundary.
				edge_edge_set.insert(edge);
			} else {
				edge_edge_set.remove(&edge);
			}
		}
	}
	for (v0, v1) in edge_edge_set {
		out[v0 as usize] = true;
		out[v1 as usize] = true;
	}
}

pub fn aabb_default() -> Aabb<f32> {
	Aabb {
		min: Vec3::broadcast(f32::MAX),
		max: Vec3::broadcast(f32::MIN),
	}
}

pub fn calc_aabb<'a>(vertices: impl IntoIterator<Item = &'a Vertex>) -> Aabb<f32> {
	let mut aabb = aabb_default();
	for v in vertices {
		aabb.min = Vec3::partial_min(aabb.min, v.position);
		aabb.max = Vec3::partial_max(aabb.max, v.position);
	}
	aabb
}

pub fn merge_spheres(a: Sphere<f32, f32>, b: Sphere<f32, f32>) -> Sphere<f32, f32> {
	let sr = a.radius.min(b.radius);
	let br = a.radius.max(b.radius);
	let len = (a.center - b.center).magnitude();
	if len + sr <= br || sr == 0.0 || len == 0.0 {
		if a.radius > b.radius {
			a
		} else {
			b
		}
	} else {
		let radius = (sr + br + len) / 2.0;
		let center = (a.center + b.center + (a.radius - b.radius) * (a.center - b.center) / len) / 2.0;
		Sphere { center, radius }
	}
}

struct TempNode {
	group: u32,
	aabb: Aabb<f32>,
	children: Vec<u32>,
}

#[derive(Default)]
pub struct BvhBuilder {
	nodes: Vec<TempNode>,
	lods: Vec<Range<u32>>,
}

impl BvhBuilder {
	fn add_lod(&mut self, offset: u32, groups: &[MeshletGroup]) {
		let start = self.nodes.len() as u32;
		self.nodes.extend(groups.iter().enumerate().map(|(i, g)| TempNode {
			group: i as u32 + offset,
			aabb: g.aabb,
			children: Vec::new(),
		}));
		let end = self.nodes.len() as u32;
		self.lods.push(start..end);
	}

	fn cost(&self, nodes: &[u32]) -> f32 {
		let ex = nodes
			.iter()
			.map(|&x| self.nodes[x as usize].aabb)
			.reduce(Aabb::union)
			.unwrap()
			.size();
		(ex.w * ex.h + ex.w * ex.d + ex.h * ex.d) * 2.0
	}

	fn optimize_splits(&self, nodes: &mut [u32], splits: [usize; 8]) {
		// Do 8 splits by repeatedly binary splitting so I can just copy someone's binary BVH code.
		for i in 0..3 {
			let parts = 1 << i;
			let count = 8 >> i;
			let half_count = count >> 1;
			let mut offset = 0;
			for p in 0..parts {
				let first = p * count;
				let mut s0 = 0;
				let mut s1 = 0;
				for i in 0..half_count {
					s0 += splits[first + i];
					s1 += splits[first + half_count + i];
				}
				let c = s0 + s1;
				let nodes = &mut nodes[offset..(offset + c)];
				offset += c;

				let mut cost = f32::MAX;
				let mut axis = 0;
				let key = |x, ax| self.nodes[x as usize].aabb.center()[ax];
				for ax in 0..3 {
					nodes.sort_unstable_by(|&x, &y| key(x, ax).partial_cmp(&key(y, ax)).unwrap());
					let (left, right) = nodes.split_at(s0);
					let c = self.cost(left) + self.cost(right);
					if c < cost {
						axis = ax;
						cost = c;
					}
				}
				if axis != 2 {
					nodes.sort_unstable_by(|&x, &y| key(x, axis).partial_cmp(&key(y, axis)).unwrap());
				}
			}
		}
	}

	fn build_temp_inner(&mut self, nodes: &mut [u32], is_lod: bool) -> u32 {
		let count = nodes.len();
		if count == 1 {
			nodes[0]
		} else if count <= 8 {
			let i = self.nodes.len();
			self.nodes.push(TempNode {
				group: u32::MAX,
				aabb: aabb_default(),
				children: nodes.iter().copied().collect(),
			});
			i as _
		} else {
			// Largest power of 8 <= `count` (where the incomplete nodes start).
			let largest_child = 1 << (count.ilog2() / 3) * 3;
			let smallest_child = largest_child >> 3;
			let extra = largest_child - smallest_child;
			let mut left = count - largest_child;
			let splits = std::array::from_fn(|_| {
				let size = left.min(extra);
				left -= size;
				smallest_child + size
			});

			if is_lod {
				self.optimize_splits(nodes, splits);
			}

			let mut offset = 0;
			let children = splits
				.into_iter()
				.map(|size| {
					let i = self.build_temp_inner(&mut nodes[offset..(offset + size)], is_lod);
					offset += size;
					i
				})
				.collect();

			let i = self.nodes.len();
			self.nodes.push(TempNode {
				group: u32::MAX,
				aabb: aabb_default(),
				children,
			});
			i as _
		}
	}

	fn build_temp(&mut self) -> u32 {
		let mut lods = Vec::with_capacity(self.lods.len());
		let l = std::mem::take(&mut self.lods);
		for lod in l {
			let mut lod: Vec<_> = lod.collect();
			let root = self.build_temp_inner(&mut lod, true);
			let node = &self.nodes[root as usize];
			if node.group != u32::MAX || node.children.len() == 8 {
				lods.push(root);
			} else {
				lods.extend(node.children.iter().copied());
			}
		}
		self.build_temp_inner(&mut lods, false)
	}

	fn build_inner(
		&self, groups: &[MeshletGroup], out: &mut Vec<BvhNode>, max_depth: &mut u32, node: u32, depth: u32,
	) -> u32 {
		*max_depth = depth.max(*max_depth);
		let node = &self.nodes[node as usize];
		let onode = out.len();
		out.push(BvhNode::default());

		for (i, &child_id) in node.children.iter().enumerate() {
			let child = &self.nodes[child_id as usize];
			if child.group != u32::MAX {
				let group = &groups[child.group as usize];
				let out = &mut out[onode];
				out.aabbs[i] = group.aabb;
				out.lod_bounds[i] = group.lod_bounds;
				out.parent_errors[i] = group.parent_error;
				out.child_offsets[i] = group.meshlets.start;
				out.child_counts[i] = group.meshlets.len() as u8;
			} else {
				let child_id = self.build_inner(groups, out, max_depth, child_id, depth + 1);
				let child = &out[child_id as usize];
				let mut aabb = aabb_default();
				let mut lod_bounds = Sphere::default();
				let mut parent_error = 0.0f32;
				for i in 0..8 {
					if child.child_counts[i] == 0 {
						break;
					}

					aabb = aabb.union(child.aabbs[i]);
					parent_error = parent_error.max(child.parent_errors[i]);
					lod_bounds = merge_spheres(lod_bounds, child.lod_bounds[i]);
				}

				let out = &mut out[onode];
				out.aabbs[i] = aabb;
				out.lod_bounds[i] = lod_bounds;
				out.parent_errors[i] = parent_error;
				out.child_offsets[i] = child_id;
				out.child_counts[i] = u8::MAX;
			}
		}

		onode as _
	}

	fn build(mut self, groups: &[MeshletGroup]) -> (Vec<BvhNode>, u32) {
		let root = self.build_temp();
		let mut out = vec![];
		let mut max_depth = 0;
		let root = self.build_inner(groups, &mut out, &mut max_depth, root, 1);
		assert_eq!(root, 0, "root must be 0");
		(out, max_depth)
	}
}
