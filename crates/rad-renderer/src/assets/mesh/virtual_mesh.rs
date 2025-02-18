use std::{array, collections::BTreeMap, io, ops::Range};

use bincode::{Decode, Encode};
use bytemuck::{Pod, Zeroable};
use meshopt::{VertexDataAdapter, VertexStream};
use metis::Graph;
use rad_core::{
	asset::{
		aref::{ARef, AssetId, LARef},
		AssetView,
		BincodeAsset,
		CookedAsset,
	},
	uuid,
	Engine,
};
use rad_graph::{
	device::Device,
	resource::{Buffer, BufferDesc, BufferType, GpuPtr, Resource},
};
use rad_world::Uuid;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use static_assertions::const_assert_eq;
use tracing::{debug_span, field, trace_span};
use vek::{Aabb, Sphere, Vec3, Vec4};

use crate::{
	assets::{
		material::{Material, MaterialView},
		mesh::{GpuVertex, Mesh, Vertex},
	},
	util::SliceWriter,
};

#[derive(Copy, Clone, Encode, Decode, Default)]
pub struct BvhNode {
	#[bincode(with_serde)]
	pub aabbs: [Aabb<f32>; 8],
	#[bincode(with_serde)]
	pub lod_bounds: [Sphere<f32, f32>; 8],
	pub parent_errors: [f32; 8],
	pub child_offsets: [u32; 8],
	pub child_counts: [u8; 8],
}

#[derive(Copy, Clone, Default, Encode, Decode)]
pub struct Meshlet {
	/// Offset of the meshlet vertex buffer relative to the parent mesh vertex buffer.
	pub vert_offset: u32,
	/// Offset of the meshlet index buffer relative to the parent mesh index buffer.
	pub index_offset: u32,
	/// Number of vertices in the meshlet.
	pub vert_count: u8,
	/// Number of triangles in the meshlet. The number of indices will be 3 times this.
	pub tri_count: u8,
	/// The AABB of the meshlet.
	#[bincode(with_serde)]
	pub aabb: Aabb<f32>,
	/// The bounds to use for LOD decisions.
	#[bincode(with_serde)]
	pub lod_bounds: Sphere<f32, f32>,
	/// The error of this meshlet.
	pub error: f32,
	/// The length of the longest edge in this meshlet.
	pub max_edge_length: f32,
}

impl Meshlet {
	fn vertices(&self) -> Range<usize> {
		(self.vert_offset as usize)..(self.vert_offset as usize + self.vert_count as usize)
	}

	fn tris(&self) -> Range<usize> {
		(self.index_offset as usize)..(self.index_offset as usize + self.tri_count as usize * 3)
	}
}

/// A virtual mesh.
#[derive(Encode, Decode)]
pub struct VirtualMesh {
	/// Vertices of the mesh.
	pub vertices: Vec<Vertex>,
	/// Indices of each meshlet - should be added to `vertex_offset`.
	pub indices: Vec<u8>,
	/// Meshlets of the mesh.
	pub meshlets: Vec<Meshlet>,
	/// The LOD BVH of the mesh.
	pub bvh: Vec<BvhNode>,
	/// The max depth of the BVH.
	pub bvh_depth: u32,
	/// The AABB of the entire mesh.
	#[bincode(with_serde)]
	pub aabb: Aabb<f32>,
	#[bincode(with_serde)]
	/// Material of the mesh.
	pub material: AssetId<Material>,
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuAabb {
	pub center: Vec3<f32>,
	pub half_extent: Vec3<f32>,
}
const_assert_eq!(std::mem::size_of::<GpuAabb>(), 24);
const_assert_eq!(std::mem::align_of::<GpuAabb>(), 4);

pub(super) fn map_aabb(aabb: Aabb<f32>) -> GpuAabb {
	GpuAabb {
		center: aabb.center(),
		half_extent: aabb.half_size().into(),
	}
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBvhNode {
	pub aabbs: [GpuAabb; 8],
	pub lod_bounds: [Vec4<f32>; 8],
	pub parent_errors: [f32; 8],
	pub child_offsets: [u32; 8],
	pub child_counts: [u8; 8],
}
const_assert_eq!(std::mem::size_of::<GpuBvhNode>(), 392);
const_assert_eq!(std::mem::align_of::<GpuBvhNode>(), 4);

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMeshlet {
	pub aabb: GpuAabb,
	pub lod_bounds: Vec4<f32>,
	pub error: f32,
	pub vertex_byte_offset: u32,
	pub index_byte_offset: u32,
	pub vertex_count: u8,
	pub triangle_count: u8,
	pub _pad: u16,
	pub max_edge_length: f32,
}
const_assert_eq!(std::mem::size_of::<GpuMeshlet>(), 60);
const_assert_eq!(std::mem::align_of::<GpuMeshlet>(), 4);

pub(super) fn map_sphere(sphere: Sphere<f32, f32>) -> Vec4<f32> { sphere.center.with_w(sphere.radius) }

impl BincodeAsset for VirtualMesh {
	type Root = Mesh;

	const UUID: Uuid = uuid!("36e2ce93-453f-4bb2-ad98-83e327a58ae6");
}

impl CookedAsset for VirtualMesh {
	type Base = Mesh;

	fn cook(mesh: &Self::Base) -> Self {
		let (_, remap) = meshopt::generate_vertex_remap_multi(
			mesh.vertices.len(),
			&[VertexStream::new_with_stride::<Vec3<f32>, _>(
				mesh.vertices.as_ptr() as *const Vec3<f32>,
				std::mem::size_of::<Vertex>(),
			)],
			Some(&mesh.indices),
		);

		let mut boundary = vec![false; mesh.vertices.len()];
		let mut meshlets = generate_meshlets(&mesh.vertices, &mesh.indices, None);

		let mut bvh = BvhBuilder::default();
		let mut simplify: Vec<_> = (0..meshlets.meshlets.len() as u32).collect();
		let mut stuck = Vec::new();
		let mut lod = 0;
		while !simplify.is_empty() {
			let s = debug_span!(
				"generating lod",
				lod,
				meshlets = simplify.len(),
				groups = field::Empty,
				min_size = field::Empty,
				avg_size = field::Empty,
				max_size = field::Empty,
			);
			let _e = s.enter();

			let groups = generate_groups(&simplify, &mut meshlets, &remap);
			simplify.clear();
			compute_boundary(&mut boundary, &meshlets, &remap, &groups);
			s.record("groups", groups.len());

			let par: Vec<_> = groups
				.into_par_iter()
				.map(|mut group| {
					if group.meshlets().count() == 1 {
						return Err(group);
					}

					let Some((indices, parent_error)) = simplify_group(&mesh.vertices, &boundary, &meshlets, &group)
					else {
						return Err(group);
					};

					group.parent_error = parent_error;
					let n_meshlets =
						generate_meshlets(&mesh.vertices, &indices, Some((group.lod_bounds, parent_error)));
					let size = group
						.meshlets()
						.map(|x| {
							let m = &meshlets.meshlets[x as usize];
							(m.vertices().len() * std::mem::size_of::<Vertex>() + m.tris().len()) as f32
						})
						.sum();
					Ok((group, n_meshlets, size))
				})
				.collect();

			let count = par.len();
			let first_group = meshlets.groups.len();

			let mut min_size = f32::MAX;
			let mut avg_size = 0.0f32;
			let mut max_size = 0.0f32;
			let mut tris = 0;
			let mut stuck_tris = 0;
			for x in par {
				match x {
					Ok((group, n_meshlets, size)) => {
						tris += n_meshlets.tris.len() / 3;
						min_size = min_size.min(size);
						avg_size += size;
						max_size = max_size.max(size);
						simplify.extend(meshlets.add(n_meshlets));
						meshlets.groups.push(group);
					},
					Err(group) => {
						stuck_tris += group
							.meshlets()
							.map(|x| meshlets.meshlets[x as usize].tri_count as usize)
							.sum::<usize>();
						stuck.push(group);
					},
				}
			}
			if count > 0 {
				s.record("min_size", min_size);
				s.record("avg_size", avg_size / count as f32);
				s.record("max_size", max_size);
			}

			if tris > stuck_tris / 3 {
				simplify.extend(stuck.drain(..).as_ref().iter().flat_map(|x| x.meshlets()));
			}

			bvh.add_lod(first_group as _, &meshlets.groups);
			lod += 1;
		}

		if !stuck.is_empty() {
			let first_group = meshlets.groups.len();
			meshlets.groups.extend(stuck);
			bvh.add_lod(first_group as _, &meshlets.groups);
		}

		let (bvh, depth) = bvh.build(&mut meshlets);
		convert_meshlets(mesh, meshlets, bvh, depth)
	}
}

#[derive(Clone)]
struct MeshletGroup {
	aabb: Aabb<f32>,
	lod_bounds: Sphere<f32, f32>,
	parent_error: f32,
	meshlets: [u32; 12],
}

impl Default for MeshletGroup {
	fn default() -> Self {
		Self {
			aabb: aabb_default(),
			lod_bounds: Sphere::default(),
			parent_error: f32::MAX,
			meshlets: [u32::MAX; 12],
		}
	}
}

impl MeshletGroup {
	fn meshlets(&self) -> impl Iterator<Item = u32> + '_ {
		self.meshlets.iter().copied().take_while(|&x| x != u32::MAX)
	}

	fn push(&mut self, meshlet: u32) {
		let mut i = 0;
		while self.meshlets[i] != u32::MAX {
			i += 1;
		}
		self.meshlets[i] = meshlet;
	}
}

struct Meshlets {
	vertex_remap: Vec<u32>,
	tris: Vec<u8>,
	groups: Vec<MeshletGroup>,
	meshlets: Vec<Meshlet>,
}

impl Meshlets {
	fn add(&mut self, other: Meshlets) -> Range<u32> {
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
			for m in g.meshlets.iter_mut() {
				*m += meshlet_offset;
			}
			g
		}));
		meshlet_offset..self.meshlets.len() as u32
	}
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

fn generate_groups(range: &[u32], meshlets: &Meshlets, remap: &[u32]) -> Vec<MeshletGroup> {
	let s = trace_span!("grouping meshlets");
	let _e = s.enter();

	// TODO: locality links
	let connections = find_connections(range, meshlets, remap);

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

	let mut out = vec![MeshletGroup::default(); group_count];
	for (i, group) in group_of.into_iter().enumerate() {
		let group = &mut out[group as usize];
		let mid = range[i];
		let m = &meshlets.meshlets[mid as usize];

		group.push(mid);
		group.aabb = group.aabb.union(m.aabb);
		group.lod_bounds = merge_spheres(group.lod_bounds, m.lod_bounds);
	}
	out
}

fn simplify_group(
	vertices: &[Vertex], locked: &[bool], meshlets: &Meshlets, group: &MeshletGroup,
) -> Option<(Vec<u32>, f32)> {
	let s = trace_span!("simplifying group");
	let _e = s.enter();

	let indices: Vec<_> = group
		.meshlets()
		.flat_map(|m| {
			let m = &meshlets.meshlets[m as usize];
			let verts = &meshlets.vertex_remap[m.vertices()];
			meshlets.tris[m.tris()].iter().map(move |&x| verts[x as usize])
		})
		.collect();

	let norm_weight = 2.0;
	let uv_weight = 0.5;
	let target = ((indices.len() / 3) / 2) * 3;

	let mut error = 0.0;
	let simplified = unsafe {
		let data: &[f32] = bytemuck::cast_slice(vertices);
		let mut res = Vec::with_capacity(indices.len());
		let count = meshopt::ffi::meshopt_simplifyWithAttributes(
			res.as_mut_ptr() as *mut _,
			indices.as_ptr() as *const _,
			indices.len(),
			data.as_ptr(),
			vertices.len(),
			std::mem::size_of::<Vertex>() as _,
			data.as_ptr().add(3),
			std::mem::size_of::<Vertex>() as _,
			[norm_weight, norm_weight, norm_weight, uv_weight, uv_weight].as_ptr(),
			5,
			locked.as_ptr() as *const _,
			target,
			f32::MAX,
			(meshopt::SimplifyOptions::Sparse | meshopt::SimplifyOptions::ErrorAbsolute).bits(),
			&mut error,
		);
		res.set_len(count);
		res
	};

	for m in group.meshlets() {
		error = error.max(meshlets.meshlets[m as usize].error);
	}

	if (simplified.len() as f32 / indices.len() as f32) < 0.55 {
		Some((simplified, error))
	} else {
		None
	}
}

// For each meshlet in `range`, all the meshlets that share edges with it, and how many edges they
// share.
// Note that the returned meshlets are indices into `range`, not the full meshlet list.
fn find_connections(range: &[u32], meshlets: &Meshlets, remap: &[u32]) -> Vec<Vec<(u32, u32)>> {
	let s = trace_span!("generating meshlet graph");
	let _e = s.enter();

	let mut shared_edges = FxHashMap::default();
	for (im, &mid) in range.iter().enumerate() {
		let m = &meshlets.meshlets[mid as usize];
		let verts = &meshlets.vertex_remap[m.vertices()];
		for i in meshlets.tris[m.tris()].chunks(3) {
			for j in 0..3 {
				let i0 = i[j] as usize;
				let i1 = i[(j + 1) % 3] as usize;
				let v0 = remap[verts[i0] as usize];
				let v1 = remap[verts[i1] as usize];
				let edge = (v0.min(v1), v0.max(v1));
				let out = shared_edges.entry(edge).or_insert(Vec::new());
				if out.last() != Some(&(im as u32)) {
					out.push(im as u32);
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
	for ((m1, m2), count) in shared_count {
		connections[m1 as usize].push((m2, count));
		connections[m2 as usize].push((m1, count));
	}
	connections
}

fn convert_meshlets(
	Mesh { vertices, material, .. }: &Mesh, meshlets: Meshlets, bvh: Vec<BvhNode>, bvh_depth: u32,
) -> VirtualMesh {
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

	VirtualMesh {
		vertices: outv,
		indices: outi,
		meshlets,
		bvh,
		bvh_depth,
		aabb,
		material: *material,
	}
}

// https://github.com/zeux/meshoptimizer/blob/master/demo/nanite.cpp
fn compute_boundary(out: &mut [bool], meshlets: &Meshlets, remap: &[u32], groups: &[MeshletGroup]) {
	let mut group_map = vec![-1; out.len()];

	for (g, group) in groups.iter().enumerate() {
		for mid in group.meshlets() {
			let m = &meshlets.meshlets[mid as usize];
			let verts = &meshlets.vertex_remap[m.vertices()];
			for &i in meshlets.tris[m.tris()].iter() {
				let v = remap[verts[i as usize] as usize] as usize;
				if group_map[v] == -1 || group_map[v] == g as i32 {
					group_map[v] = g as i32;
				} else {
					group_map[v] = -2;
				}
			}
		}
	}

	for (v, &r) in out.iter_mut().zip(remap) {
		*v = group_map[r as usize] == -2;
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
		self.nodes
			.extend(groups.iter().enumerate().skip(offset as _).map(|(i, g)| TempNode {
				group: i as u32,
				aabb: g.aabb,
				children: Vec::new(),
			}));
		let end = self.nodes.len() as u32;
		if start != end {
			self.lods.push(start..end);
		}
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
		for lod in std::mem::take(&mut self.lods) {
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
				out.child_offsets[i] = group.meshlets[0];
				out.child_counts[i] = group.meshlets[1] as _;
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

	fn mark_reachable(&self, out: &[BvhNode], reachable: &mut [bool], node: u32) {
		let node = &out[node as usize];
		for i in 0..8 {
			if node.child_counts[i] == u8::MAX {
				self.mark_reachable(out, reachable, node.child_offsets[i]);
			} else {
				for m in 0..node.child_counts[i] as u32 {
					reachable[(m + node.child_offsets[i]) as usize] = true;
				}
			}
		}
	}

	fn build(mut self, meshlets: &mut Meshlets) -> (Vec<BvhNode>, u32) {
		let s = trace_span!("build bvh");
		let _e = s.enter();

		// The BVH requires group meshlets to be contiguous, so remap them first.
		let mut remap = Vec::with_capacity(meshlets.meshlets.len());
		for group in meshlets.groups.iter_mut() {
			let first = remap.len();
			let count = group.meshlets().count();
			for m in group.meshlets() {
				remap.push(meshlets.meshlets[m as usize]);
			}
			group.meshlets[0] = first as u32;
			group.meshlets[1] = count as u32;
		}
		meshlets.meshlets = remap;

		let mut out = vec![];
		let mut max_depth = 0;

		if self.nodes.len() == 1 {
			let mut o = BvhNode::default();
			let group = &meshlets.groups[0];
			o.aabbs[0] = group.aabb;
			o.lod_bounds[0] = group.lod_bounds;
			o.parent_errors[0] = group.parent_error;
			o.child_offsets[0] = group.meshlets[0];
			o.child_counts[0] = group.meshlets[1] as _;
			out.push(o);
			max_depth = 1;
		} else {
			let root = self.build_temp();
			let root = self.build_inner(&meshlets.groups, &mut out, &mut max_depth, root, 1);
			assert_eq!(root, 0, "root must be 0");
		}

		let mut reachable = vec![false; meshlets.meshlets.len()];
		self.mark_reachable(&out, &mut reachable, 0);
		assert!(reachable.iter().all(|&x| x), "all meshlets must be reachable");

		(out, max_depth)
	}
}

pub struct VirtualMeshView {
	buffer: Buffer,
	bvh_depth: u32,
	aabb: Aabb<f32>,
	material: LARef<MaterialView>,
}

impl VirtualMeshView {
	pub fn bvh_depth(&self) -> u32 { self.bvh_depth }

	pub fn aabb(&self) -> Aabb<f32> { self.aabb }

	pub fn gpu_aabb(&self) -> GpuAabb { map_aabb(self.aabb) }

	pub fn gpu_ptr(&self) -> GpuPtr<u8> { self.buffer.ptr() }

	pub fn material(&self) -> &LARef<MaterialView> { &self.material }
}

impl AssetView for VirtualMeshView {
	type Base = VirtualMesh;
	type Ctx = ();

	fn load(_: &'static Self::Ctx, m: Self::Base) -> Result<Self, io::Error> {
		let device: &Device = Engine::get().global();
		// TODO: fips.
		let name = "virtual mesh";

		let s = trace_span!("loading virtual mesh", name = name);
		let _e = s.enter();

		let bvh_byte_offset = 0;
		let bvh_byte_len = (m.bvh.len() * std::mem::size_of::<GpuBvhNode>()) as u64;
		let meshlet_byte_offset = bvh_byte_offset + bvh_byte_len;
		let meshlet_byte_len = (m.meshlets.len() * std::mem::size_of::<GpuMeshlet>()) as u64;
		let vertex_byte_offset = meshlet_byte_offset + meshlet_byte_len;
		let vertex_byte_len = (m.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
		let index_byte_offset = vertex_byte_offset + vertex_byte_len;
		let index_byte_len = (m.indices.len() * std::mem::size_of::<u8>()) as u64;
		let size = index_byte_offset + index_byte_len;

		let buffer = Buffer::create(
			device,
			BufferDesc {
				name: &format!("{name} buffer"),
				size,
				ty: BufferType::Gpu,
			},
		)
		.map_err(|x| io::Error::new(io::ErrorKind::Other, format!("failed to create mesh buffer: {:?}", x)))?;
		let mut writer = SliceWriter::new(unsafe { buffer.data().as_mut() });

		for node in m.bvh {
			writer.write(GpuBvhNode {
				aabbs: node.aabbs.map(map_aabb),
				lod_bounds: node.lod_bounds.map(map_sphere),
				parent_errors: node.parent_errors,
				child_offsets: array::from_fn(|i| {
					if node.child_counts[i] == u8::MAX {
						bvh_byte_offset as u32 + node.child_offsets[i] * std::mem::size_of::<GpuBvhNode>() as u32
					} else {
						meshlet_byte_offset as u32 + node.child_offsets[i] * std::mem::size_of::<GpuMeshlet>() as u32
					}
				}),
				child_counts: node.child_counts,
			});
		}

		for me in m.meshlets.iter() {
			writer.write(GpuMeshlet {
				aabb: map_aabb(me.aabb),
				lod_bounds: map_sphere(me.lod_bounds),
				error: me.error,
				vertex_byte_offset: vertex_byte_offset as u32
					+ (me.vert_offset * std::mem::size_of::<GpuVertex>() as u32),
				index_byte_offset: index_byte_offset as u32 + (me.index_offset * std::mem::size_of::<u8>() as u32),
				vertex_count: me.vert_count,
				triangle_count: me.tri_count,
				_pad: 0,
				max_edge_length: me.max_edge_length,
			});
		}

		writer.write_slice(&m.vertices);
		writer.write_slice(&m.indices);

		Ok(Self {
			buffer,
			bvh_depth: m.bvh_depth,
			aabb: m.aabb,
			material: ARef::loaded(m.material)?,
		})
	}
}
