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

pub fn aabb_to_sphere(aabb: Aabb<f32>) -> Sphere<f32, f32> {
	Sphere {
		center: aabb.center(),
		radius: aabb.half_size().magnitude(),
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
		&self, groups: &[MeshletGroup], out: &mut Vec<BvhNode>, temp: &mut Vec<Aabb<f32>>, max_depth: &mut u32,
		node: u32, depth: u32,
	) -> u32 {
		*max_depth = depth.max(*max_depth);
		let node = &self.nodes[node as usize];
		let onode = out.len();
		out.push(BvhNode::default());
		temp.push(aabb_default());

		let mut lod_aabb = aabb_default();
		for (i, &child_id) in node.children.iter().enumerate() {
			let child = &self.nodes[child_id as usize];
			if child.group != u32::MAX {
				let group = &groups[child.group as usize];
				let out = &mut out[onode];
				out.aabbs[i] = group.aabb;
				out.lod_bounds[i] = aabb_to_sphere(group.lod_bounds);
				out.parent_errors[i] = group.parent_error;
				out.child_offsets[i] = group.meshlets.start;
				out.child_counts[i] = group.meshlets.len() as u8;
				lod_aabb = lod_aabb.union(group.lod_bounds);
			} else {
				let child_id = self.build_inner(groups, out, temp, max_depth, child_id, depth + 1);
				let child = &out[child_id as usize];
				let mut aabb = aabb_default();
				let mut child_lod_aabb = aabb_default();
				let mut parent_error = 0.0f32;
				for i in 0..8 {
					if child.child_counts[i] == 0 {
						break;
					}

					aabb = aabb.union(child.aabbs[i]);
					child_lod_aabb = child_lod_aabb.union(temp[child_id as usize]);
					parent_error = parent_error.max(child.parent_errors[i]);
				}
				let out = &mut out[onode];
				out.aabbs[i] = aabb;
				out.lod_bounds[i] = aabb_to_sphere(child_lod_aabb);
				out.parent_errors[i] = parent_error;
				out.child_offsets[i] = child_id;
				out.child_counts[i] = u8::MAX;
				lod_aabb = lod_aabb.union(child_lod_aabb);
			}
		}
		temp[onode] = lod_aabb;

		onode as _
	}

	pub fn build(mut self, groups: &[MeshletGroup]) -> (Vec<BvhNode>, u32) {
		let root = self.build_temp();
		let mut out = vec![];
		let mut temp = vec![];
		let mut max_depth = 0;
		let root = self.build_inner(groups, &mut out, &mut temp, &mut max_depth, root, 1);
		assert_eq!(root, 0, "root must be 0");
		(out, max_depth)
	}
}
