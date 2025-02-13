module meshlet;

import graph;
import asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex2D<f32> hzb;
	Sampler hzb_sampler;
	CandidateMeshletQueue queue;
	MeshletQueue render;
	CullStats* stats;
	u64 frame;
	u32x2 res;
}

[vk::push_constant]
PushConstants Constants;

void write(bool visible, bool hw, NodePointer p) {
	// note: these must be branches because `push` assumes all active lanes are calling it.
	if (visible) {
		if (hw) {
			if (Constants.render.push_hw(p))
				wave_atomic_or(Constants.stats->overflow, 1);
		} else {
			if (Constants.render.push_sw(p))
				wave_atomic_or(Constants.stats->overflow, 1);
		}
	} else if (EARLY) {
		if (Constants.queue.push_late(p, 1))
			wave_atomic_or(Constants.stats->overflow, 1);
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	let count = Constants.queue.count();
	if (id == 0)
		get_stats(Constants.stats)->candidate_meshlets = count;
	if (id >= count)
		return;

	let p = Constants.queue.get(id);
	let instance = &Constants.instances[p.instance];
	let meshlet = instance->meshlet(p.node_offset);
	let c = Cull(Constants.camera, instance, Constants.frame, Constants.res, Constants.hzb, Constants.hzb_sampler);
	let aabb = meshlet->aabb;
	let render = c.should_render(meshlet->lod_bounds, meshlet->error);
	if (c.in_frustum(aabb) && render) {
		// let hw = c.hw_or_sw(meshlet.aabb, meshlet.max_edge_length);
		write(c.unoccluded(aabb), true, { p.instance, p.node_offset });
	}
}
