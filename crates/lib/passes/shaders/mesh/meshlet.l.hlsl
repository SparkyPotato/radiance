#include "cull.l.hlsl"

struct PushConstants {
	Buf<Instance> instances;
	Buf<Camera> camera;
	Tex2D hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	MeshletQueue early;
	MeshletQueue late;
	uint2 res;
};

PUSH PushConstants Constants;

#ifdef EARLY
float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mul(Constants.camera.load(1).view_proj, transform);
}

void write(bool visible, MeshletPointer p) {
	if (visible) {
		Constants.early.push(p);
	} else {
		Constants.late.push(p);
	}
}
#else
float4x4 occ_camera(float4x4 mvp, float4x4 transform) {
	return mvp;
}

void write(bool visible, MeshletPointer p) {
	if (visible) {
		Constants.late.push(p);
	}
}
#endif

[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len()) return;

	BvhNodePointer p = Constants.read.get(id);
	Instance instance = Constants.instances.load(p.instance);
	Meshlet meshlet = instance.mesh.load<Meshlet>(p.node, 0);
	Camera camera = Constants.camera.load(0);
	Camera prev_camera = Constants.camera.load(1);
	
	float4x4 transform = instance.get_transform();
	float4x4 mv = mul(camera.view, transform);
	float4x4 mvp = mul(camera.view_proj, transform);
	float4x4 omvp = occ_camera(mvp, transform);

	Cull c = Cull::init(mv, mvp, Constants.res, camera.h);
	OccCull oc = OccCull::init(omvp, Constants.res, Constants.hzb, Constants.hzb_sampler);

	Aabb aabb = meshlet.aabb;
	float4 lod_bounds = meshlet.lod_bounds;
	f32 error = meshlet.error;
	if (c.is_perceptible(lod_bounds, error) || c.frustum_cull(aabb)) return;

	MeshletPointer ret;
	ret.instance = p.instance;
	ret.meshlet_offset = p.node;
	write(!oc.cull(aabb), ret);
}
