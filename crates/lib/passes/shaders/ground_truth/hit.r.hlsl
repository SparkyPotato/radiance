#include "common.l.hlsl"

[shader("closesthit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attrs) {
	Instance instance = Constants.instances.load(InstanceID());
	float4x3 mat = ObjectToWorld4x3();
	float3 bary = float3(1.0 - attrs.barycentrics.x - attrs.barycentrics.y, attrs.barycentrics.x, attrs.barycentrics.y);

	u32 ioff = instance.index_byte_offset;
	u32 prim = PrimitiveIndex();
	u32 i0 = instance.raw_mesh.load<u32>(ioff, prim * 3 + 0);
	u32 i1 = instance.raw_mesh.load<u32>(ioff, prim * 3 + 1);
	u32 i2 = instance.raw_mesh.load<u32>(ioff, prim * 3 + 2);

	float3 p0 = float3(instance.raw_mesh.load<Pos>(0, i0));
	float3 p1 = float3(instance.raw_mesh.load<Pos>(0, i1));
	float3 p2 = float3(instance.raw_mesh.load<Pos>(0, i2));
	float3 position = mul(mat, float4(bary.x * p0 + bary.y * p1 + bary.z * p2, 1.f));

	u32 voff = instance.mesh.load<Meshlet>(instance.submesh_count * sizeof(Submesh), 0).vertex_offset;
	Vertex v0 = instance.mesh.load<Vertex>(voff, i0);
	Vertex v1 = instance.mesh.load<Vertex>(voff, i1);
	Vertex v2 = instance.mesh.load<Vertex>(voff, i2);

	float3 n0 = (float3(v0.normal) / 32767.f) * 2.f - 1.f;
	float3 n1 = (float3(v1.normal) / 32767.f) * 2.f - 1.f;
	float3 n2 = (float3(v2.normal) / 32767.f) * 2.f - 1.f;
	float3 normal = normalize(mul(mat, bary.x * n0 + bary.y * n1 + bary.z * n2));

	float3 light = normalize(float3(1.f, 1.f, 1.f));
	float3 color = 1.f;
	float3 ret = color * saturate(dot(light, normal)) / 3.1415926f;

	RayDesc ray;
	ray.Origin = position;
	ray.Direction = light;
	ray.TMin = 0.1f;
	ray.TMax = 10000.f;

	ShadowPayload s;
	s.unshadowed = false;
	TraceRay(
		ASes[Constants.as.index], 	
		RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
		0xff, 
		0, 
		0, 
		1, 
		ray, 
		s
	);

	payload.value = float4(ret * s.unshadowed + 0.05f, 1.f);
}

