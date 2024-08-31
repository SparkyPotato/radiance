module mesh;

import graph;
import passes.asset;
import cull;
import visbuffer;

struct VertexOutput {
	f32x4 position : SV_Position;
};

struct PrimitiveOutput {
	[vk::location(0)]
	u32 data : VisBuffer;
}

struct PushConstants {
	Instance* instances;
	Camera* camera;
	MeshletQueue early;
	MeshletQueue late;
	VisBufferTex output;
}

[vk::push_constant]
PushConstants Constants;

MeshletPointer get(u32 gid) {
	if (EARLY)
		return Constants.early.get(gid);
	else
		return Constants.late.get(gid);
}

u32 meshlet_id(u32 gid) {
	if (EARLY)
		return gid;
	else
		return Constants.early.len() + gid;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex, out vertices VertexOutput vertices[64],
		  out indices u32x3 triangles[124], out primitives PrimitiveOutput visbuffer[124]) {
	let p = get(gid);
	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.meshlet_offset);
	let camera = Constants.camera[0];
	let mid = meshlet_id(gid);

	let vert_count = meshlet.vertex_count;
	let tri_count = meshlet.tri_count;
	SetMeshOutputCounts(vert_count, tri_count);
	let mvp = mul(camera.view_proj, instance.get_transform());
	let dim = Constants.output.size();

	if (gtid < vert_count) {
		let v = meshlet.vertex(instance.mesh, gtid);
		vertices[gtid].position = transform_vertex(mvp, v).clip;
	}

	if (gtid < tri_count) {
		let i = meshlet.tri(instance.mesh, gtid);
		triangles[gtid] = i;
		VisBufferData data = { mid, gtid };
		visbuffer[gtid].data = data.encode();
	}
}
