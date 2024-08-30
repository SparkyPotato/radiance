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

extern static const bool EARLY;

f32x4x4 occ_camera(f32x4x4 mvp, f32x4x4 transform) {
	if (EARLY)
		return mul(Constants.camera[1].view_proj, transform);
	else
		return mvp;
}

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
	let camera = Constants.camera[0];

	let transform = instance.get_transform();
	let mv = mul(camera.view, transform);
	let mvp = mul(camera.view_proj, transform);
	let omvp = occ_camera(mvp, transform);

	let c = Cull(mv, mvp, Constants.res, camera.near, camera.h);
	let oc = OccCull(omvp, Constants.res, camera.near, Constants.hzb, Constants.hzb_sampler);

	if (c.frustum_cull(meshlet.aabb) || c.is_perceptible(meshlet.lod_bounds, meshlet.error))
		return;

	write(!oc.cull(meshlet.aabb), { p.instance, p.node });
}
