#include "radiance-passes/asset/data.l.hlsl"
#include "cull.l.hlsl"
#include "visbuffer.l.hlsl"

struct VertexOutput {
	float4 position: SV_Position;
};

struct PrimitiveOutput {
	[[vk::location(0)]] u32 data: VisBuffer;
};

struct PushConstants {
	Buf<Instance> instances;
	Buf<Camera> camera;
	MeshletQueue early;
	MeshletQueue late;
};

PUSH PushConstants Constants;

#ifdef EARLY
MeshletPointer get(u32 gid) {
	return Constants.early.get(gid);
}

u32 meshlet_id(u32 gid) {
	return gid;
}
#else
MeshletPointer get(u32 gid) {
	return Constants.late.get(gid);
}

u32 meshlet_id(u32 gid) {
	return Constants.early.len() + gid;
}
#endif

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void main(
	u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
	out vertices VertexOutput vertices[64],
	out indices uint3 triangles[124],
	out primitives PrimitiveOutput visbuffer[124]
) {
	MeshletPointer p = get(gid);
	Instance instance = Constants.instances.load(p.instance);
	Meshlet meshlet = instance.mesh.load<Meshlet>(p.meshlet_offset, 0);
	Camera camera = Constants.camera.load(0);

	u32 vert_count = meshlet.vertex_count();
	u32 tri_count = meshlet.tri_count();
	SetMeshOutputCounts(vert_count, tri_count);

	float4x4 mvp = mul(camera.view_proj, instance.get_transform());

	if (gtid < vert_count) {
		Vertex vertex = meshlet.vertex(instance.mesh, gtid);
		vertices[gtid].position = mul(mvp, float4(vertex.position, 1.f));
	}

	[unroll]
	for (u32 i = 0; i < 2; i++) {
		u32 t = min(gtid + i * 64, tri_count);
		u32 indices = meshlet.tri(instance.mesh, t);
		triangles[t] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

		VisBufferData data = { meshlet_id(gid), t };
		visbuffer[t].data = data.encode();
	}
}
