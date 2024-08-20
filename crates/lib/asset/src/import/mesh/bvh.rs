use std::ops::Range;

use vek::{Aabb, Sphere, Vec3};

use crate::{
	import::mesh::MeshletGroup,
	mesh::{BvhNode, Vertex},
};

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
	if len + sr < br || sr == 0.0 || len == 0.0 {
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
	pub fn add_lod(&mut self, offset: u32, groups: &[MeshletGroup]) {
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
		&self, groups: &[MeshletGroup], out: &mut Vec<BvhNode>, at: usize, max_depth: &mut u32, node: u32, depth: u32,
	) {
		let node = &self.nodes[node as usize];
		out[at] = if node.group != u32::MAX {
			*max_depth = depth.max(*max_depth);
			let group = &groups[node.group as usize];
			let child_count = (group.meshlets.len() as u8) | (1 << 7);
			BvhNode {
				aabb: group.aabb,
				lod_bounds: group.lod_bounds,
				parent_error: group.parent_error,
				children_offset: group.meshlets.start,
				child_count,
			}
		} else {
			let base = out.len();
			let count = node.children.len();
			out.extend(std::iter::repeat(BvhNode::default()).take(count));
			for (i, &c) in node.children.iter().enumerate() {
				self.build_inner(groups, out, base + i, max_depth, c, depth + 1)
			}
			let mut aabb = aabb_default();
			let mut lod_bounds = Sphere::default();
			let mut parent_error = 0.0f32;
			for n in &out[base..(base + count)] {
				aabb = aabb.union(n.aabb);
				lod_bounds = merge_spheres(lod_bounds, n.lod_bounds);
				parent_error = parent_error.max(n.parent_error);
			}
			BvhNode {
				aabb,
				lod_bounds,
				parent_error,
				children_offset: base as u32,
				child_count: count as u8,
			}
		};
	}

	pub fn build(mut self, groups: &[MeshletGroup]) -> (Vec<BvhNode>, u32) {
		let root = self.build_temp();
		let mut out = vec![BvhNode::default()];
		let mut max_depth = 0;
		self.build_inner(groups, &mut out, 0, &mut max_depth, root, 1);
		(out, max_depth)
	}
}
