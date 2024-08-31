module meshlet;

import graph;
import passes.asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	MeshletQueue early;
	MeshletQueue late;
	u32x2 res;
}

[vk::push_constant]
PushConstants Constants;

void write(bool visible, MeshletPointer p) {
	if (EARLY) {
		if (visible) {
			Constants.early.push(p);
		} else {
			Constants.late.push(p);
		}
	} else {
		if (visible) {
			Constants.late.push(p);
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len())
		return;

	let p = Constants.read.get(id);
	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node);
	let c = Cull(Constants.camera, instance.get_transform(), Constants.res, Constants.hzb, Constants.hzb_sampler);
	if (c.frustum_cull(meshlet.aabb) || c.is_perceptible(meshlet.lod_bounds, meshlet.error))
		return;
	write(!c.oc_cull(meshlet.aabb), { p.instance, p.node });
}
