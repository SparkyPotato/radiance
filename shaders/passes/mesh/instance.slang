module instance;

import graph;
import passes.asset;
import cull;

struct LateInstances {
	u32 count;
	u32x3 dispatch;
	u32 instances[];
}

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue early;
	BvhQueue late;
	LateInstances* late_instances;
	u32 instance_count;
	u32x2 res;
}

[vk::push_constant]
PushConstants Constants;

u32 instance_count() {
	if (EARLY)
		return Constants.instance_count;
	else
		return Constants.late_instances->count;
}

u32 instance_id(u32 id) {
	if (EARLY)
		return id;
	else
		return Constants.late_instances->instances[id];
}

void write(bool visible, u32 id) {
	if (EARLY) {
		if (visible) {
			Constants.early.push( { id, 0 });
		} else {
			let off = atomic_add(Constants.late_instances->count, 1);
			Constants.late_instances->instances[off] = id;
			if ((off & 63) == 0)
				atomic_add(Constants.late_instances->dispatch.x, 1);
		}
	} else {
		if (visible) {
			Constants.late.push( { id, 0 });
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 tid: SV_DispatchThreadID) {
	if (tid >= instance_count())
		return;

	let id = instance_id(tid);
	let instance = Constants.instances[id];
	let c = Cull(Constants.camera, instance.get_transform(), Constants.res, Constants.hzb, Constants.hzb_sampler);
	if (c.frustum_cull(instance.aabb))
		return;
	write(!c.oc_cull(instance.aabb), id);
}
