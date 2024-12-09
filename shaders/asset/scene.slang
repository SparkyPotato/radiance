module scene_update;

import graph;
import asset;

enum UpdateType {
	Add,
	Move,
	ChangeMesh,
	ChangeTransform,
}

struct SceneUpdate {
	u32 instance;
	UpdateType ty;
	NewInstance data;
}

struct PushConstants {
	Instance* instances;
	SceneUpdate* updates;
	u64 frame;
	u32 count;
}

[vk::push_constant]
PushConstants Constants;

[shader("compute")]
[numthreads(64, 1, 1)]
void update(u32 id: SV_DispatchThreadID) {
	if (id >= Constants.count)
		return;

	let update = Constants.updates[id];
	let instance = &Constants.instances[update.instance];
	switch (update.ty) {
		case UpdateType.Add: {
			*instance = Instance(update.data);
			break;
		}
		case UpdateType.Move: {
			*instance = Constants.instances[asuint(update.data.transform.translation.x)];
			break;
		}
		case UpdateType.ChangeMesh: {
			instance->mesh = update.data.mesh;
			break;
		}
		case UpdateType.ChangeTransform: {
			instance->update_frame = Constants.frame;
			instance->last_updated_transform = instance->transform;
			instance->transform = update.data.transform;
			break;
		}
	}
}
