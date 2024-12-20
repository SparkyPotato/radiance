module scene;

import graph;
import asset;

struct Update {
	Transform transform;  // for lights, position = radiance, scale = pos_or_dir
	Aabb aabb;
	u8* mesh;
	Vertex* raw_mesh;
	Material* material;
	u64 as;
	u32 raw_vertex_count;  // move => index, add light => ty
	u32 raw_tri_count;

	Instance to_instance() {
		Instance i;
		i.transform = this.transform;
		i.last_updated_transform = this.transform;
		i.aabb = this.aabb;
		i.mesh = this.mesh;
		i.raw_mesh = this.raw_mesh;
		i.material = this.material;
		i.raw_vertex_count = this.raw_vertex_count;
		i.raw_tri_count = this.raw_tri_count;
		i.update_frame = 0;
		return i;
	}
}

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
	AddLight,
	MoveLight,
}

struct SceneUpdate {
	u32 instance;
	UpdateType ty;
	Update data;
}

struct PushConstants {
	Instance* instances;
	VkAccelerationStructureInstanceKHR* as_instances;
	Light* lights;
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
	switch (update.ty) {
		case UpdateType.Add: {
			let instance = &Constants.instances[update.instance];
			let as_instance = &Constants.as_instances[update.instance];
			*instance = update.data.to_instance();
			as_instance->transform = instance->transform.vk_mat();
			as_instance->instanceCustomIndexAndMask = (0xff << 24) | id;
			as_instance->instanceShaderBindingTableRecordOffsetAndFlags = 0;
			as_instance->accelerationStructureReference = update.data.as;
			break;
		}
		case UpdateType.Move: {
			let instance = &Constants.instances[update.instance];
			let as_instance = &Constants.as_instances[update.instance];
			let ind = update.data.raw_vertex_count;
			*instance = Constants.instances[ind];
			*as_instance = Constants.as_instances[ind];
			break;
		}
		case UpdateType.ChangeMesh: {
			let instance = &Constants.instances[update.instance];
			let as_instance = &Constants.as_instances[update.instance];
			instance->mesh = update.data.mesh;
			instance->raw_mesh = update.data.raw_mesh;
			instance->raw_vertex_count = update.data.raw_vertex_count;
			instance->raw_tri_count = update.data.raw_tri_count;
			instance->material = update.data.material;
			as_instance->accelerationStructureReference = update.data.as;
			break;
		}
		case UpdateType.ChangeTransform: {
			let instance = &Constants.instances[update.instance];
			let as_instance = &Constants.as_instances[update.instance];
			instance->update_frame = Constants.frame;
			instance->last_updated_transform = instance->transform;
			instance->transform = update.data.transform;
			as_instance->transform = instance->transform.vk_mat();
			break;
		}
		case UpdateType.AddLight: {
			let light = &Constants.lights[update.instance];
			light->ty = LightType(update.data.raw_vertex_count);
			light->radiance = update.data.transform.translation;
			light->pos_or_dir = update.data.transform.scale;
			break;
		}
		case UpdateType.MoveLight: {
			let light = &Constants.lights[update.instance];
			let li = update.data.raw_vertex_count;
			*light = Constants.lights[li];
			break;
		}
	}
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
