module scene_update;

import graph;
import asset;

struct VkAccelerationStructureInstanceKHR {
    f32 transform[3][4];
    u32 instanceCustomIndexAndMask;
    u32 instanceShaderBindingTableRecordOffsetAndFlags;
    u64 accelerationStructureReference;
}

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
	VkAccelerationStructureInstanceKHR* as_instances;
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
	let as_instance = &Constants.as_instances[update.instance];
	switch (update.ty) {
		case UpdateType.Add: {
			*instance = Instance(update.data);
			as_instance->transform = instance->transform.vk_mat();
			as_instance->instanceCustomIndexAndMask = (id << 24) | 0xff;
			as_instance->instanceShaderBindingTableRecordOffsetAndFlags = 0;
			as_instance->accelerationStructureReference = update.data.as;
			break;
		}
		case UpdateType.Move: {
			let ind = asuint(update.data.transform.translation.x);
			*instance = Constants.instances[ind];
			*as_instance = Constants.as_instances[ind];
			break;
		}
		case UpdateType.ChangeMesh: {
			instance->mesh = update.data.mesh;
			as_instance->accelerationStructureReference = update.data.as;
			break;
		}
		case UpdateType.ChangeTransform: {
			instance->update_frame = Constants.frame;
			instance->last_updated_transform = instance->transform;
			instance->transform = update.data.transform;
			as_instance->transform = instance->transform.vk_mat();
			break;
		}
	}
}

extension Transform {
	f32[4][3] vk_mat() {
		let m = this.mat();
		// clang-format off
		return{
			f32[4](m._m00, m._m01, m._m02, m._m03),
			f32[4](m._m10, m._m11, m._m12, m._m13),
			f32[4](m._m20, m._m21, m._m22, m._m23)
		};
		// clang-format on
	}
}
