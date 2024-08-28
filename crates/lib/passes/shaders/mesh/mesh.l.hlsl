#include "radiance-passes/asset/data.l.hlsl"
#include "cull.l.hlsl"
#include "visbuffer.l.hlsl"

struct VertexOutput {
	float4 position: SV_Position;
};

struct PrimitiveOutput {
	bool cull : SV_CullPrimitive;
	[[vk::location(0)]] u32 data : VisBuffer;
};

struct PushConstants {
	Buf<Instance> instances;
	Buf<Camera> camera;
	MeshletQueue early;
	MeshletQueue late;
	VisBufferTex output;
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

#define SW_THRESH 8

groupshared float3 Positions[64];
groupshared u32 SWCount;
groupshared uint3 SWTriangles[124];

template<typename T>
T min3(T x, T y, T z) {
	return min(x, min(y, z));
}

template<typename T>
T max3(T x, T y, T z) {
	return max(x, max(y, z));
}

f32 edge_fn(float2 a, float2 b, float2 c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(
	u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex,
	out vertices VertexOutput vertices[64],
	out indices uint3 triangles[124],
	out primitives PrimitiveOutput visbuffer[124]
) {
	if (gtid == 0) {
		SWCount = 0;
	}

	MeshletPointer p = get(gid);
	Instance instance = Constants.instances.load(p.instance);
	Meshlet meshlet = instance.mesh.load<Meshlet>(p.meshlet_offset, 0);
	Camera camera = Constants.camera.load(0);
	u32 mid = meshlet_id(gid);

	u32 vert_count = meshlet.vertex_count();
	u32 tri_count = meshlet.tri_count();
	SetMeshOutputCounts(vert_count, tri_count);
	float4x4 mvp = mul(camera.view_proj, instance.get_transform());
	float2 dim = Constants.output.dimensions();

	if (gtid < vert_count) {
		Vertex vertex = meshlet.vertex(instance.mesh, gtid);
		float4 clip = mul(mvp, float4(vertex.position, 1.f));
		float3 ndc = clip.xyz / clip.w;
		float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
		Positions[gtid] = float3(uv * dim, ndc.z);
		vertices[gtid].position = clip;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid < tri_count) {
		u32 indices = meshlet.tri(instance.mesh, gtid);
		uint3 i = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;
		triangles[gtid] = i;
		VisBufferData data = { mid, gtid };
		visbuffer[gtid].data = data.encode();
		float3 v0 = Positions[i.x];
		float3 v1 = Positions[i.y];
		float3 v2 = Positions[i.z];
		uint2 minv = max(uint2(min3(v0, v1, v2).xy), uint2(0, 0));
		uint2 maxv = min(uint2(ceil(max3(v0, v1, v2).xy)), uint2(dim) - 1);
		if (all(maxv - minv <= SW_THRESH)) {
			u32 index;
			InterlockedAdd(SWCount, 1, index);
			SWTriangles[index] = i;
			visbuffer[gtid].cull = true;
		} else {
			visbuffer[gtid].cull = false;
		}
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid >= SWCount) return;

	uint3 t = SWTriangles[gtid];
	float3 v0 = Positions[t.x];
	float3 v1 = Positions[t.y];
	float3 v2 = Positions[t.z];
	VisBufferData data = { mid, gtid };
	u32 write = data.encode();

	uint2 minv = max(uint2(min3(v0, v1, v2).xy), uint2(0, 0));
	uint2 maxv = min(uint2(ceil(max3(v0, v1, v2).xy)), uint2(dim) - 1);
	if (any(minv > maxv)) return;

	float3 w_x = float3(v1.y - v2.y, v2.y - v0.y, v0.y - v1.y);
	float3 w_y = float3(v2.x - v1.x, v0.x - v2.x, v1.x - v0.x);
	f32 par_area = edge_fn(v0.xy, v1.xy, v2.xy);
	float3 vz = float3(v0.z, v1.z, v2.z) / par_area;
	f32 z_x = dot(vz, w_x);
	f32 z_y = dot(vz, w_y);

	float2 start = minv + 0.5f;
	float3 w_row = float3(
		edge_fn(v1.xy, v2.xy, start),
		edge_fn(v2.xy, v0.xy, start),
		edge_fn(v0.xy, v1.xy, start)
	);
	f32 z_row = dot(vz, w_row);
	for (u32 y = minv.y; y <= maxv.y; y++) {
		float3 w = w_row;
		f32 z = z_row;
		for (u32 x = minv.x; x <= maxv.x; x++) {
			if (min3(w.x, w.y, w.z) >= 0.f) Constants.output.write(uint2(x, y), min3(v0, v1, v2).z, write, true);

			w += w_x;
			z += z_x;
		}
		w_row += w_y;
		z_row += z_y;
	}
}
