module meshlet;

import graph;
import asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	BvhQueue late;
	MeshletQueue hw;
	MeshletQueue sw;
	u32x2 res;
	u32 len;
}

[vk::push_constant]
PushConstants Constants;

void write(bool visible, bool hw, NodePointer p) {
	let len = Constants.len;
	// note: these must be branches because `push` assumes all active lanes are calling it.
	if (visible) {
		if (hw) {
			Constants.hw.push(len, p);
		} else {
			Constants.sw.push(len, p);
		}
	} else if (EARLY) {
		Constants.late.push_meshlet(len, p);
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len())
		return;

	let p = Constants.read.get(id);
	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node_offset);
	let c = Cull(Constants.camera, instance.transform, Constants.res, Constants.hzb, Constants.hzb_sampler);
	let render = c.should_render(meshlet.lod_bounds, meshlet.error, meshlet.max_edge_length);
	if (c.in_frustum(meshlet.aabb) && render.x)
		write(c.unoccluded(meshlet.aabb), render.y, { p.instance, p.node_offset });
}
