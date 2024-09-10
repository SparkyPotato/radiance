module scene_update;

import asset;

struct TransformUpdate {
	u32 instance;
	Transform transform;
}

struct PushConstants {
	Instance* instances;
	TransformUpdate* updates;
	u32 count;
}

[vk::push_constant]
PushConstants Constants;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.count)
		return;

	let update = Constants.updates[id];
	Constants.instances[update.instance].transform = update.transform;
}
