module scene;

import graph;
import asset;

struct VkAccelerationStructureInstanceKHR {
	f32 transform[3][4];
	u32 instanceCustomIndexAndMask;
	u32 instanceShaderBindingTableRecordOffsetAndFlags;
	u64 accelerationStructureReference;
}

extension Transform {
	f32[4][3] vk_mat() {
		let m = this.mat();
		// clang-format off
		return {
			f32[4](m._m00, m._m01, m._m02, m._m03),
			f32[4](m._m10, m._m11, m._m12, m._m13),
			f32[4](m._m20, m._m21, m._m22, m._m23)
		};
		// clang-format on
	}
}

struct RtUpdate {
	u32 index;
	u32 _pad;
	u64 as;
	RtInstance instance;
}

struct RtConstants {
	RtInstance* instances;
	VkAccelerationStructureInstanceKHR* as_instances;
	RtUpdate* updates;
	u32 count;
}

[vk::push_constant]
RtConstants RConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void update_rt(u32 id: SV_DispatchThreadID) {
	if (id >= RConstants.count)
		return;

	let update = RConstants.updates[id];
	RConstants.instances[update.index] = update.instance;
	RConstants.as_instances[update.index] =
		VkAccelerationStructureInstanceKHR(update.instance.transform.vk_mat(), 0xff << 24, 0, update.as);
}

struct VirtualUpdate {
	u32 index;
	u32 _pad;
	Instance instance;
}

struct VirtualConstants {
	Instance* instances;
	VirtualUpdate* updates;
	u32 count;
}

[vk::push_constant]
VirtualConstants VConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void update_virtual(u32 id: SV_DispatchThreadID) {
	if (id >= VConstants.count)
		return;

	let update = VConstants.updates[id];
	VConstants.instances[update.index] = update.instance;
}
