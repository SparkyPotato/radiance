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
		Constants.next.push( { id, 0 });
	} else if (EARLY) {
		let rel = WavePrefixCountBits(true);
		let total = WaveActiveCountBits(true);
		u32 pos;
		if (WaveIsFirstLane())
			pos = atomic_add(Constants.late_instances->count, total);
		pos = WaveReadLaneFirst(pos) + rel;
		Constants.late_instances->instances[pos] = id;
		if ((pos & 63) == 0)
			atomic_add(Constants.late_instances->dispatch.x, 1);
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 tid: SV_DispatchThreadID) {
	if (tid >= instance_count())
		return;

	let id = instance_id(tid);
	let instance = Constants.instances[id];
	let c = Cull(Constants.camera, instance.transform, Constants.res, Constants.hzb, Constants.hzb_sampler);
	if (c.in_frustum(instance.aabb))
		write(c.unoccluded(instance.aabb), id);
}
