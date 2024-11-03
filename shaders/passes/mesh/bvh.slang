module bvh;

import graph;
import asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue queue;
	BvhQueue late;
	CandidateMeshletQueue meshlet;
	u32x2 res;
	bool ping;
};

[vk::push_constant]
PushConstants Constants;

void write(bool visible, u8 count, NodePointer p) {
	// note: these must be branches because `push` assumes all active lanes are calling it.
	if (visible) {
		if (count == 255) {
			if (Constants.ping)
				Constants.queue.push_back(p);
			else
				Constants.queue.push_front(p);
		} else {
			Constants.meshlet.push_early(p, count);
		}
	} else if (EARLY) {
		if (count == 255) {
			Constants.late.push_front(p);
		} else {
			Constants.meshlet.push_late(p, count);
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID, u32 gtid: SV_GroupIndex) {
	let node = id >> 3;
	let subnode = id & 7;

	// Always keep the check *before* `read.done()`.
	if (Constants.ping) {
		let should_exit = node >= Constants.queue.front_count();
		if (gtid == 0)
			Constants.queue.done_front();
		if (should_exit)
			return;
	} else {
		let should_exit = node >= Constants.queue.back_count();
		if (gtid == 0)
			Constants.queue.done_back();
		if (should_exit)
			return;
	}

	var p = Constants.ping ? Constants.queue.get_front(node) : Constants.queue.get_back(node);
	let instance = Constants.instances[p.instance];
	let n = instance.node(p.node_offset);
	let c = Cull(Constants.camera, instance.transform, Constants.res, Constants.hzb, Constants.hzb_sampler);

	let aabb = n.aabbs[subnode];
	let lod_bounds = n.lod_bounds[subnode];
	let parent_error = n.parent_errors[subnode];
	if (c.in_frustum(aabb) && c.should_visit_bvh(lod_bounds, parent_error)) {
		p.node_offset = n.child_offsets[subnode];
		write(c.unoccluded(aabb), n.child_counts[subnode], p);
	}
}
