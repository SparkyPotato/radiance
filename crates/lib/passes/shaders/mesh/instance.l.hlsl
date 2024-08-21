#include "cull.l.hlsl"

struct PushConstants {
	Buf<Instance> instances;
	Buf<Camera> camera;
	Tex2D hzb;
	Sampler hzb_sampler;
	BvhQueue early;
	BvhQueue late;
	Buf<u32> late_instances;
	u32 instance_count;
	uint2 res;
};

PUSH PushConstants Constants;

#ifdef EARLY
u32 instance_count() {
	return Constants.instance_count;
}

u32 instance_id(u32 id) {
	return id;
}

float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mul(Constants.camera.load(1).view_proj, transform);
}

bool frustum_cull(Cull c, Aabb aabb) {
	return c.frustum_cull(aabb);
}

void write(bool visible, u32 id) {
	if (visible) {
		BvhNodePointer p;
		p.instance = id;
		p.node = 0;
		Constants.early.push(p);
	} else {
		u32 off = Constants.late_instances.atomic_add(0, 1);
		Constants.late_instances.store(4 + off, id);
		if ((off & 63) == 0) Constants.late_instances.atomic_add(1, 1);
	}
}
#else
u32 instance_count() {
	return Constants.late_instances.load(0);
}

u32 instance_id(u32 id) {
	return Constants.late_instances.load(4 + id);
}

float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mvp;
}

bool frustum_cull(Cull c, Aabb aabb) {
	return false;
}

void write(bool visible, u32 id) {
	if (visible) {
		BvhNodePointer p;
		p.instance = id;
		p.node = 0;
		Constants.late.push(p);
	}
}
#endif

[numthreads(64, 1, 1)]
void main(u32 tid: SV_DispatchThreadID) {
	if (tid >= instance_count()) return;

	u32 id = instance_id(tid);
	Instance instance = Constants.instances.load(id);
	Camera camera = Constants.camera.load(0);
	
	float4x4 transform = instance.get_transform();
	float4x4 mv = mul(camera.view, transform);
	float4x4 mvp = mul(camera.view_proj, transform);
	float4x4 omvp = occ_camera(mvp, transform);

	Cull c = Cull::init(mv, mvp, Constants.res, camera.h);
	OccCull oc = OccCull::init(omvp, Constants.res, camera.near, Constants.hzb, Constants.hzb_sampler);
	Aabb aabb = instance.aabb;
	if (frustum_cull(c, aabb)) return;

	write(!oc.cull(aabb), id);
}
