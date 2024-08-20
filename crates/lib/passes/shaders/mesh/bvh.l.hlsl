#include "cull.l.hlsl"

struct PushConstants {
	Buf<Instance> instances;
	Buf<Camera> camera;
	Tex2D hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	BvhQueue next;
	BvhQueue meshlet;
	BvhQueue late;
	BvhQueue late_meshlet;
	uint2 res;
};

PUSH PushConstants Constants;

#ifdef EARLY
float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mul(Constants.camera.load(1).view_proj, transform);
}

void write(bool visible, u32 count, BvhNodePointer p) {
	bool is_meshlet = (count >> 7) == 1;
	count = count & 0b01111111;
	if (visible) {
		if (is_meshlet) {
			Constants.meshlet.push(p, count, sizeof(Meshlet));
		} else {
			Constants.next.push(p, count, sizeof(BvhNode));
		}
	} else {
		if (is_meshlet) {
			Constants.late_meshlet.push(p, count, sizeof(Meshlet));
		} else {
			Constants.late.push(p, count, sizeof(BvhNode));
		}
	}
}
#else
float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mvp;
}

void write(bool visible, u32 count, BvhNodePointer p) {
	bool is_meshlet = (count >> 7) == 1;
	count = count & 0b01111111;
	if (visible) {
		if (is_meshlet) {
			Constants.late_meshlet.push(p, count, sizeof(Meshlet));
		} else {
			Constants.next.push(p, count, sizeof(BvhNode));
		}
	}
}
#endif

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len()) return;

	BvhNodePointer p = Constants.read.get(id);
	Instance instance = Constants.instances.load(p.instance);
	BvhNode node = instance.mesh.load<BvhNode>(p.node, 0);
	Camera camera = Constants.camera.load(0);
	Camera prev_camera = Constants.camera.load(1);
	
	float4x4 transform = instance.get_transform();
	float4x4 mv = mul(camera.view, transform);
	float4x4 mvp = mul(camera.view_proj, transform);
	float4x4 omvp = occ_camera(mvp, transform);

	Cull c = Cull::init(mv, mvp, Constants.res, camera.h);
	OccCull oc = OccCull::init(omvp, Constants.res, Constants.hzb, Constants.hzb_sampler);
	Aabb aabb = node.aabb;
	float4 lod_bounds = node.lod_bounds;
	f32 parent_error = node.parent_error;
	if (c.is_imperceptible(lod_bounds, parent_error) || c.frustum_cull(aabb)) return;

	p.node = node.children_offset;
	write(!oc.cull(aabb), node.child_count, p);
}
