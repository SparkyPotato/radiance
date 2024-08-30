module bvh;

import graph;
import passes.asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	BvhQueue next;
	BvhQueue meshlet;
	BvhQueue late;
	BvhQueue late_meshlet;
	u32x2 res;
};

[vk::push_constant]
PushConstants Constants;

extern static const bool EARLY;

f32x4x4 occ_camera(f32x4x4 mvp, f32x4x4 transform) {
	if (EARLY)
		return mul(Constants.camera[1].view_proj, transform);
	else
		return mvp;
}

void write(bool visible, u32 count, BvhNodePointer p) {
	let is_meshlet = (count >> 7) == 1;
	count = count & 0b01111111;
	if (EARLY) {
		if (visible) {
			if (is_meshlet) {
				Constants.meshlet.push(p, count, sizeof(Meshlet));
			} else {
				Constants.next.push(p, count, sizeof(BvhNode));
			}
		} else {
			if (is_meshlet) {
				Constants.late_meshlet.push(p, count, sizeof(Meshlet));
			} else {
				Constants.late.push(p, count, sizeof(BvhNode));
			}
		}
	} else {
		if (visible) {
			if (is_meshlet) {
				Constants.late_meshlet.push(p, count, sizeof(Meshlet));
			} else {
				Constants.next.push(p, count, sizeof(BvhNode));
			}
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len())
		return;

	var p = Constants.read.get(id);
	let instance = Constants.instances[p.instance];
	let node = instance.node(p.node);
	let camera = Constants.camera[0];

	let transform = instance.get_transform();
	let mv = mul(camera.view, transform);
	let mvp = mul(camera.view_proj, transform);
	let omvp = occ_camera(mvp, transform);

	let c = Cull(mv, mvp, Constants.res, camera.near, camera.h);
	let oc = OccCull(omvp, Constants.res, camera.near, Constants.hzb, Constants.hzb_sampler);
	if (c.frustum_cull(node.aabb) || c.is_imperceptible(node.lod_bounds, node.parent_error))
		return;

	p.node = node.children_offset;
	write(!oc.cull(node.aabb), node.child_count, p);
}
