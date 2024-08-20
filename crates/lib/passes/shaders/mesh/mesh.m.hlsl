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
	MeshletQueue read;
};

PUSH PushConstants Constants;

[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(
	u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
	out vertices VertexOutput vertices[128],
	out indices uint3 triangles[124],
	out primitives PrimitiveOutput visbuffer[124]
) {
	MeshletPointer p = Constants.read.get(gid);
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

	if (gtid < tri_count) {
		u32 indices = meshlet.tri(instance.mesh, gtid);
		triangles[gtid] = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

		VisBufferData data = { gid, gtid }; // TODO: fix
		visbuffer[gtid].data = data.encode();
	}
}
