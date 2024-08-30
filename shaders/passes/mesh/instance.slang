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

extern static const bool EARLY;

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

f32x4x4 occ_camera(f32x4x4 mvp, f32x4x4 transform) {
	if (EARLY)
		return mul(Constants.camera[1].view_proj, transform);
	else
		return mvp;
}

bool frustum_cull(Cull c, Aabb aabb) {
	if (EARLY)
		return c.frustum_cull(aabb);
	else
		return false;
}

void write(bool visible, u32 id) {
	if (EARLY) {
		if (visible) {
			Constants.early.push({ id, 0 });
		} else {
			let off = atomic_add(Constants.late_instances->count, 1);
			Constants.late_instances->instances[off] = id;
			if ((off & 63) == 0)
				atomic_add(Constants.late_instances->dispatch.x, 1);
		}
	} else {
		if (visible) {
			Constants.late.push({ id, 0 });
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
	let camera = Constants.camera[0];

	let transform = instance.get_transform();
	let mv = mul(camera.view, transform);
	let mvp = mul(camera.view_proj, transform);
	let omvp = occ_camera(mvp, transform);

	let c = Cull(mv, mvp, Constants.res, camera.near, camera.h);
	let oc = OccCull(omvp, Constants.res, camera.near, Constants.hzb, Constants.hzb_sampler);
	if (frustum_cull(c, instance.aabb))
		return;
	write(!oc.cull(instance.aabb), id);
}
