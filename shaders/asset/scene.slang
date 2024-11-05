module scene_update;

import graph;
import asset;

struct TransformUpdate {
	u32 instance;
	Transform transform;
}

struct PushConstants {
	Instance* instances;
	TransformUpdate* updates;
	u64 frame;
	u32 count;
}

[vk::push_constant]
PushConstants Constants;

[shader("compute")]
[numthreads(64, 1, 1)]
void tick(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.count)
		return;

	let update = Constants.updates[id];
	let instance = &Constants.instances[update.instance];
	instance->update_frame = Constants.frame;
	instance->last_updated_transform = instance->transform;
	instance->transform = update.transform;
}
