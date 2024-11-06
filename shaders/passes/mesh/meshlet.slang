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
			Constants.render.push_hw(p);
		} else {
			Constants.render.push_sw(p);
		}
	} else if (EARLY) {
		Constants.queue.push_late(p, 1);
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
	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node_offset);
	let c = Cull(Constants.camera, instance, Constants.frame, Constants.res, 
			     Constants.hzb, Constants.hzb_sampler);
	let render = c.should_render(meshlet.lod_bounds, meshlet.error, meshlet.max_edge_length);
	if (c.in_frustum(meshlet.aabb) && render.x)
		write(c.unoccluded(meshlet.aabb), render.y, { p.instance, p.node_offset });
}
