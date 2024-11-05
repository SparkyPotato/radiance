module instance;

import graph;
import asset;
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
	BvhQueue next;
	LateInstances* late_instances;
	CullStats* stats;
	u64 frame;
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
	if (visible) {
		Constants.next.push_front( { id, 0 });
	} else if (EARLY) {
		let pos = wave_atomic_inc(Constants.late_instances->count);
		Constants.late_instances->instances[pos] = id;
		let req = (pos + 1 + 63) >> 6;
		if ((pos & 63) == 0)
			wave_atomic_max(Constants.late_instances->dispatch.x, req);
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 tid: SV_DispatchThreadID) {
	let count = instance_count();
	if (tid == 0)
		get_stats(Constants.stats)->instances = count;
	if (tid >= count)
		return;

	let id = instance_id(tid);
	let instance = Constants.instances[id];
	let c = Cull(Constants.camera, instance, Constants.frame, Constants.res, 
			     Constants.hzb, Constants.hzb_sampler);
	if (c.in_frustum(instance.aabb))
		write(c.unoccluded(instance.aabb), id);
}
