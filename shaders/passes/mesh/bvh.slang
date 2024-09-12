module bvh;

import graph;
import asset;
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
	u32 len;
};

[vk::push_constant]
PushConstants Constants;

void write(bool visible, u8 count, NodePointer p) {
	let len = Constants.len;
	// note: these must be branches because `push` assumes all active lanes are calling it.
	if (visible) {
		if (count == 255) {
			Constants.next.push(len, p);
		} else {
			Constants.meshlet.push_meshlets(len, p, count);
		}
	} else if (EARLY) {
		if (count == 255) {
			Constants.late.push(len, p);
		} else {
			Constants.late_meshlet.push_meshlets(len, p, count);
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupIndex) {
	let node = id >> 3;
	let subnode = id & 7;

	let should_exit = node >= Constants.read.len();
	if (gtid == 0)
		Constants.read.done();
	if (should_exit)
		return;

	var p = Constants.read.get(node);
	let instance = Constants.instances[p.instance];
	let n = instance.node(p.node_offset);
	let c = Cull(Constants.camera, instance.transform, Constants.res, Constants.hzb, Constants.hzb_sampler);

	let aabb = n.aabbs[subnode];
	let lod_bounds = n.lod_bounds[subnode];
	let parent_error = n.parent_errors[subnode];
	if (c.frustum_cull(aabb) || c.should_render(lod_bounds, parent_error))
		return;
	p.node_offset = n.child_offsets[subnode];
	write(!c.oc_cull(aabb), n.child_counts[subnode], p);
}
