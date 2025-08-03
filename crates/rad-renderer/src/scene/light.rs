use std::f32::consts::PI;

use bytemuck::NoUninit;
use rad_graph::{
	graph::{BufferDesc, BufferUsage, Frame, PassBuilder, PassContext, Res},
	resource::{BufferHandle, GpuPtr},
	sync::Shader,
};
use rad_world::{
	TickStage,
	World,
	bevy_ecs::{
		entity::Entity,
		schedule::IntoSystemConfigs,
		system::{Commands, Query, ResMut, Resource},
	},
	tick::Tick,
	transform::Transform,
};
use vek::{Aabb, Vec3};

use crate::{
	assets::mesh::NormalCone,
	components::light::{LightComponent, LightType},
	scene::{GpuScene, rt_scene::KnownRtInstances, should_scene_sync},
};

#[repr(C)]
#[derive(Copy, Clone, NoUninit)]
pub struct GpuLightScene {
	buf: GpuPtr<GpuLightTreeNode>,
	emissive_paths: GpuPtr<u32>,
}

#[derive(Copy, Clone)]
pub struct LightScene {
	buf: Res<BufferHandle>,
	emissive_paths: Res<BufferHandle>,
	pub sun_radiance: Vec3<f32>,
	pub sun_dir: Vec3<f32>,
}

impl LightScene {
	pub fn reference(&self, pass: &mut PassBuilder, shader: Shader) {
		pass.reference(self.buf, BufferUsage::read(shader));
		pass.reference(self.emissive_paths, BufferUsage::read(shader));
	}

	pub fn get(&self, pass: &mut PassContext) -> GpuLightScene {
		GpuLightScene {
			buf: pass.get(self.buf).ptr(),
			emissive_paths: pass.get(self.emissive_paths).ptr(),
		}
	}
}

impl GpuScene for LightScene {
	type In = ();
	type Res = LightSceneData;

	fn add_to_world(world: &mut World, tick: &mut Tick) {
		world.insert_resource(LightSceneData::new());
		tick.add_systems(TickStage::Render, sync_lights.run_if(should_scene_sync::<Self>));
	}

	fn update<'pass>(frame: &mut Frame<'pass, '_>, data: &'pass mut LightSceneData, _: &Self::In) -> Self {
		let mut pass = frame.pass("upload light scene");
		let buf = pass.resource(
			BufferDesc::upload(std::mem::size_of::<GpuLightTreeNode>() as u64 * data.light_tree.len() as u64),
			BufferUsage::none(),
		);
		let emissive_paths = pass.resource(
			BufferDesc::upload(std::mem::size_of::<u32>() as u64 * data.emissive_paths.len() as u64),
			BufferUsage::none(),
		);
		let sun_radiance = data.sun_radiance;
		let sun_dir = data.sun_dir;
		pass.build(move |mut pass| {
			pass.write(buf, 0, &data.light_tree);
			pass.write(emissive_paths, 0, &data.emissive_paths);
		});
		Self {
			buf,
			emissive_paths,
			sun_radiance,
			sun_dir,
		}
	}
}

// TODO: global the pipeline.
pub struct LightSceneData {
	sun_radiance: Vec3<f32>,
	sun_dir: Vec3<f32>,
	light_tree: Vec<GpuLightTreeNode>,
	emissive_paths: Vec<u32>,
}
impl Resource for LightSceneData {}

impl LightSceneData {
	fn new() -> Self {
		Self {
			sun_radiance: Vec3::zero(),
			sun_dir: -Vec3::unit_z(),
			light_tree: Vec::new(),
			emissive_paths: Vec::new(),
		}
	}
}

#[derive(Copy, Clone, Debug)]
struct BuildLight {
	aabb: Aabb<f32>,
	power: Vec3<f32>,
	cone: NormalCone,
	// If nodes, both are indices into the list.
	// If leaves, left is u32::MAX, and right is 0 if point light else mesh index + 1.
	left: u32,
	right: u32,
}

fn transform_light(t: Transform, mut light: BuildLight) -> BuildLight {
	let mat = t.into_matrix();
	let a = light.aabb;
	let corners = [
		Vec3::new(a.min.x, a.min.y, a.min.z),
		Vec3::new(a.max.x, a.min.y, a.min.z),
		Vec3::new(a.min.x, a.max.y, a.min.z),
		Vec3::new(a.max.x, a.max.y, a.min.z),
		Vec3::new(a.min.x, a.min.y, a.max.z),
		Vec3::new(a.max.x, a.min.y, a.max.z),
		Vec3::new(a.min.x, a.max.y, a.max.z),
		Vec3::new(a.max.x, a.max.y, a.max.z),
	];
	let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
	let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
	for c in corners {
		let c = (mat * c.with_w(1.0)).xyz();
		min = Vec3::partial_min(min, c);
		max = Vec3::partial_max(max, c);
	}
	light.aabb = Aabb { min, max };
	for i in light.aabb.min.into_iter().chain(light.aabb.max.into_iter()) {
		if i.is_nan() {
			panic!("NaN in light AABB: {light:?}");
		}
	}
	light.cone.axis = (mat * light.cone.axis.with_w(0.0)).xyz();
	light
}

fn merge_lights(left: BuildLight, right: BuildLight) -> BuildLight {
	BuildLight {
		aabb: Aabb {
			min: Vec3::partial_min(left.aabb.min, right.aabb.min),
			max: Vec3::partial_max(left.aabb.max, right.aabb.max),
		},
		power: left.power + right.power,
		cone: left.cone.merge(right.cone),
		left: left.left,
		right: right.right,
	}
}

// SAOH, from [Importance Sampling of Many Lights with Adaptive Tree Splitting, Kulla and Conty,
// 2018]
fn m_omega(cone: NormalCone) -> f32 {
	let theta_o = cone.theta_o;
	let theta_e = cone.theta_e;
	let theta_w = (theta_o + theta_e).min(PI);
	let (sin_o, cos_o) = theta_o.sin_cos();
	let w_2 = theta_w * 2.0;
	2.0 * PI * (1.0 - cos_o) + 0.5 * PI * (w_2 * sin_o - (theta_o - w_2).cos() - 2.0 * theta_o * sin_o + cos_o)
}

fn surface_area(extents: Vec3<f32>) -> f32 {
	2.0 * (extents.x * extents.y + extents.x * extents.z + extents.y * extents.z)
}

fn luminance(x: Vec3<f32>) -> f32 { 0.2126 * x.x + 0.7152 * x.y + 0.0722 * x.z }

fn saoh(curr: BuildLight, left: BuildLight, right: BuildLight, axis: u32) -> f32 {
	let extents = curr.aabb.max - curr.aabb.min;
	let k_r = extents.reduce_partial_max() / extents[axis as usize];

	let curr_omega = m_omega(curr.cone);
	let curr_sa = surface_area(extents);

	let left_omega = m_omega(left.cone);
	let left_sa = surface_area(left.aabb.max - left.aabb.min);
	let right_omega = m_omega(right.cone);
	let right_sa = surface_area(right.aabb.max - right.aabb.min);

	k_r * (luminance(left.power) * left_omega * left_sa + luminance(right.power) * right_omega * right_sa)
		/ (curr_omega * curr_sa)
}

fn build_bvh(nodes: &mut Vec<BuildLight>, indices: &mut [u32]) -> u32 {
	let count = indices.len();
	if count == 1 {
		indices[0]
	} else if count == 2 {
		let i = nodes.len();
		let left = indices[0];
		let right = indices[1];
		nodes.push(BuildLight {
			left,
			right,
			..merge_lights(nodes[left as usize], nodes[right as usize])
		});
		i as _
	} else {
		let merged = indices
			.iter()
			.map(|&i| nodes[i as usize])
			.reduce(merge_lights)
			.unwrap();
		let p_40 = (count as f32 * 0.4) as usize;
		let p_60 = (count as f32 * 0.6) as usize;
		let mut cost = f32::INFINITY;
		let mut axis = 0;
		let mut split = 0;
		let key = |x, ax| nodes[x as usize].aabb.center()[ax];
		for ax in 0..3 {
			indices.sort_unstable_by(|&x, &y| key(x, ax).partial_cmp(&key(y, ax)).unwrap());
			for s in p_40..=p_60 {
				let (left, right) = indices.split_at_mut(s);
				let left_merged = left
					.iter()
					.map(|&i| nodes[i as usize])
					.reduce(merge_lights)
					.unwrap();
				let right_merged = right
					.iter()
					.map(|&i| nodes[i as usize])
					.reduce(merge_lights)
					.unwrap();
				let c = saoh(merged, left_merged, right_merged, ax as _);
				if c < cost {
					cost = c;
					axis = ax;
					split = s;
				}
			}
		}
		if axis != 2 {
			indices.sort_unstable_by(|&x, &y| key(x, axis).partial_cmp(&key(y, axis)).unwrap());
		}

		let (left, right) = indices.split_at_mut(split);
		let left = build_bvh(nodes, left);
		let right = build_bvh(nodes, right);
		let i = nodes.len() as u32;
		nodes.push(BuildLight {
			left,
			right,
			..merge_lights(nodes[left as usize], nodes[right as usize])
		});
		i as _
	}
}

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
pub struct GpuSgLight {
	pos: Vec3<f32>,
	variance: f32,
	intensity: Vec3<f32>,
	axis: Vec3<f32>,
	sharpness: f32,
}

#[derive(Copy, Clone, Default, NoUninit)]
#[repr(C)]
pub struct GpuLightTreeNode {
	left: GpuSgLight,
	right: GpuSgLight,
	// If indices are u32::MAX, then the relevant node is a point light.
	// If indices have their MSB set, then the node is an emissive.
	left_index: u32,
	right_index: u32,
}

// [Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting, AMD, 2024]
fn axis_to_vmf(axis: Vec3<f32>) -> (Vec3<f32>, f32) {
	let len = axis.magnitude();
	let len2 = len * len;
	let len3 = len2 * len;
	let sharpness = (3.0 * len - len3) / (1.0 - len2);
	let axis = if len > 0.0 { axis / len } else { Vec3::zero() };
	(axis, sharpness)
}

fn expm1_over_x(x: f32) -> f32 {
	let u = x.exp();
	if u == 1.0 {
		return 1.0;
	}
	let y = u - 1.0;
	if x.abs() < 1.0 {
		return y / u.ln();
	}

	y / x
}

fn sg_integral(sharpness: f32) -> f32 { 4.0 * PI * expm1_over_x(-2.0 * sharpness) }

fn power_to_intensity(power: Vec3<f32>, sharpness: f32) -> Vec3<f32> { power / (2.0 * PI * sg_integral(sharpness)) }

fn leaf_sg_light(light: BuildLight, meshes: &[Vec3<f32>]) -> GpuSgLight {
	let r = light.aabb.half_size().magnitude();
	debug_assert_eq!(light.left, u32::MAX);
	let (axis, sharpness) = if light.right == 0 {
		(light.cone.axis, 0.0)
	} else {
		axis_to_vmf(meshes[light.right as usize - 1])
	};
	GpuSgLight {
		pos: light.aabb.center(),
		variance: 0.5 * r * r,
		intensity: power_to_intensity(light.power, sharpness),
		axis,
		sharpness,
	}
}

fn handle_child(
	out: &mut Vec<GpuLightTreeNode>, bvh: &[BuildLight], meshes: &[Vec3<f32>], node: u32,
) -> (GpuSgLight, u32) {
	let n = bvh[node as usize];
	if n.left == u32::MAX {
		(
			leaf_sg_light(n, meshes),
			if n.right == 0 {
				u32::MAX
			} else {
				(n.right - 1) | (1 << 31)
			},
		)
	} else {
		let child_id = build_gpu_bvh(out, bvh, meshes, node);
		let child = &out[child_id as usize];

		let left_lum = luminance(bvh[n.left as usize].power);
		let right_lum = luminance(bvh[n.right as usize].power);
		let w_left = left_lum / (left_lum + right_lum);
		let w_right = right_lum / (left_lum + right_lum);

		let variance = child.left.variance * w_left
			+ child.right.variance * w_right
			+ w_left * w_right * (child.left.pos - child.right.pos).magnitude_squared();

		let axis_avg = child.left.axis * w_left + child.right.axis * w_right;
		let (axis, sharpness) = axis_to_vmf(axis_avg);

		(
			GpuSgLight {
				pos: child.left.pos * w_left + child.right.pos * w_right,
				variance,
				intensity: power_to_intensity(n.power, sharpness),
				axis,
				sharpness,
			},
			child_id,
		)
	}
}

fn build_gpu_bvh(out: &mut Vec<GpuLightTreeNode>, bvh: &[BuildLight], meshes: &[Vec3<f32>], node: u32) -> u32 {
	let us_index = out.len();
	let us = &bvh[node as usize];
	out.push(GpuLightTreeNode::default());
	let left = handle_child(out, bvh, meshes, us.left);
	let right = handle_child(out, bvh, meshes, us.right);
	let out = &mut out[us_index];
	(out.left, out.left_index) = left;
	(out.right, out.right_index) = right;
	us_index as _
}

fn emissive_paths(out: &mut Vec<u32>, bvh: &[GpuLightTreeNode], node: u32, depth: u32, path: u32) {
	let node = &bvh[node as usize];
	let left_path = path;
	let right_path = path | (1 << depth);
	if (node.left_index >> 31) & 1 == 1 {
		if node.left_index != u32::MAX {
			let i = node.left_index & !(1 << 31);
			out[i as usize] = left_path;
		}
	} else {
		emissive_paths(out, bvh, node.left_index, depth + 1, left_path);
	}
	if (node.right_index >> 31) & 1 == 1 {
		if node.right_index != u32::MAX {
			let i = node.right_index & !(1 << 31);
			out[i as usize] = right_path;
		}
	} else {
		emissive_paths(out, bvh, node.right_index, depth + 1, right_path);
	}
}

fn sync_lights(
	mut r: ResMut<LightSceneData>, cmd: Commands, punctual: Query<(Entity, &Transform, &LightComponent)>,
	emissive: Query<(Entity, &Transform, &KnownRtInstances)>,
) {
	let mut lights = Vec::new();
	let mut vmf_normals = Vec::new();
	for (e, t, l) in punctual.iter() {
		match l.ty {
			LightType::Point => lights.push(BuildLight {
				aabb: Aabb {
					min: t.position,
					max: t.position,
				},
				power: l.radiance * 4.0 * PI,
				cone: NormalCone {
					axis: Vec3::unit_x(),
					theta_o: PI,
					theta_e: PI / 2.0,
				},
				left: u32::MAX,
				right: 0,
			}),
			LightType::Directional => {
				r.sun_radiance = l.radiance;
				r.sun_dir = t.rotation * -Vec3::unit_z();
			},
		}
	}
	for (e, t, m) in emissive.iter() {
		for (&i, m, mat) in m.0.iter().map(|(i, v)| (i, v, &v.material)) {
			if mat.average_emissive.iter().all(|&x| x <= 0.0) {
				continue;
			}

			if vmf_normals.len() <= i as usize {
				vmf_normals.resize(i as usize + 1, Vec3::zero());
			}

			vmf_normals[i as usize] = m.normal_average;
			lights.push(transform_light(
				*t,
				BuildLight {
					aabb: m.aabb,
					power: mat.average_emissive * m.area * PI,
					cone: m.normal_cone,
					left: u32::MAX,
					right: i + 1,
				},
			));
		}
	}

	let r = &mut *r;

	r.light_tree.clear();
	if lights.is_empty() {
		r.light_tree.push(GpuLightTreeNode {
			left: GpuSgLight::default(),
			right: GpuSgLight::default(),
			left_index: u32::MAX,
			right_index: u32::MAX,
		});
		return;
	}

	let mut indices: Vec<_> = (0..lights.len() as u32).collect();
	let root = build_bvh(&mut lights, &mut indices);

	if lights.len() == 1 {
		let light = lights[0];
		r.light_tree.push(GpuLightTreeNode {
			left: leaf_sg_light(light, &vmf_normals),
			right: GpuSgLight::default(),
			left_index: if light.right == 0 {
				u32::MAX
			} else {
				(light.right - 1) | (1 << 31)
			},
			right_index: u32::MAX,
		});
	} else {
		let root = build_gpu_bvh(&mut r.light_tree, &lights, &vmf_normals, root);
		assert_eq!(root, 0, "Root of the light BVH should be 0");
	}

	r.emissive_paths.clear();
	r.emissive_paths.resize(vmf_normals.len(), 0);
	emissive_paths(&mut r.emissive_paths, &r.light_tree, 0, 0, 0);
}
