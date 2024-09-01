module bvh;

import graph;
import passes.asset;
import cull;

struct PushConstants {
	Instance* instances;
	Camera* camera;
	Tex<f32, D2> hzb;
	Sampler hzb_sampler;
	BvhQueue read;
	BvhQueue next;
	BvhQueue meshlet;
	BvhQueue late;
	BvhQueue late_meshlet;
	u32x2 res;
	u32 len;
};

[vk::push_constant]
PushConstants Constants;

void write(bool visible, u32 count, BvhNodePointer p) {
	let is_meshlet = (count >> 7) == 1;
	count = count & 0b01111111;
	let len = Constants.len;
	if (EARLY) {
		if (visible) {
			if (is_meshlet) {
				Constants.meshlet.push(len, p, count, sizeof(Meshlet));
			} else {
				Constants.next.push(len, p, count, sizeof(BvhNode));
			}
		} else {
			if (is_meshlet) {
				Constants.late_meshlet.push(len, p, count, sizeof(Meshlet));
			} else {
				Constants.late.push(len, p, count, sizeof(BvhNode));
			}
		}
	} else {
		if (visible) {
			if (is_meshlet) {
				Constants.late_meshlet.push(len, p, count, sizeof(Meshlet));
			} else {
				Constants.next.push(len, p, count, sizeof(BvhNode));
			}
		}
	}
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.read.len())
		return;

	var p = Constants.read.get(id);
	let instance = Constants.instances[p.instance];
	let node = instance.node(p.node);
	let c = Cull(Constants.camera, instance.get_transform(), Constants.res, Constants.hzb, Constants.hzb_sampler);
	if (c.frustum_cull(node.aabb) || c.is_imperceptible(node.lod_bounds, node.parent_error))
		return;
	p.node = node.children_offset;
	write(!c.oc_cull(node.aabb), node.child_count, p);
}
