#include "common.l.hlsl"

u32 hash(u32 a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

[shader("closesthit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attrs) {
	Instance instance = Constants.instances.load(InstanceID());
	float3x4 smat = ObjectToWorld3x4();
	float4x4 tmat = {
		smat._m00, smat._m01, smat._m02, smat._m03,
		smat._m10, smat._m11, smat._m12, smat._m13,
		smat._m20, smat._m21, smat._m22, smat._m23,
		0.f,       0.f,       0.f,       0.f
	};
	float3 bary = float3(1.0 - attrs.barycentrics.x - attrs.barycentrics.y, attrs.barycentrics.x, attrs.barycentrics.y);
	u32 meshlet_index = GeometryIndex();
	Meshlet meshlet = instance.mesh.load<Meshlet>(sizeof(Submesh) * instance.submesh_count, meshlet_index);

	u32 mat_index = instance.mesh.load<Submesh>(0, meshlet.submesh).mat_index;
	Material mat = Constants.materials.load(mat_index);

	u32 prim = PrimitiveIndex();
	u32 indices = instance.mesh.load<u32>(meshlet.index_offset, prim);
	uint3 i = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

	Vertex v0 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.x);
	Vertex v1 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.y);
	Vertex v2 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.z);

	float3 min = float3(meshlet.aabb_min);
	float3 extent = float3(meshlet.aabb_extent);
	float3 p0 = float3(min + extent * float3(v0.position) / 65535.f);
	float3 p1 = float3(min + extent * float3(v1.position) / 65535.f);
	float3 p2 = float3(min + extent * float3(v2.position) / 65535.f);
	float3 position = mul(tmat, float4(bary.x * p0 + bary.y * p1 + bary.z * p2, 1.f)).xyz;

	float3 n0 = (float3(v0.normal) / 32767.f) * 2.f - 1.f;
	float3 n1 = (float3(v1.normal) / 32767.f) * 2.f - 1.f;
	float3 n2 = (float3(v2.normal) / 32767.f) * 2.f - 1.f;
	float3 normal = normalize(mul(tmat, float4(bary.x * n0 + bary.y * n1 + bary.z * n2, 0.f))).xyz;

	float2 u0 = float2(v0.uv) / 65535.f;
	float2 u1 = float2(v1.uv) / 65535.f;
	float2 u2 = float2(v2.uv) / 65535.f;
	float2 uv = bary.x * u0 + bary.y * u1 + bary.z * u2;

	// float3 light = normalize(float3(1.f, 1.f, 1.f));
	// float3 color = mat.base_color.load(mat.base_color.pixel_of_uv(uv));
	// float3 ret = color; // * saturate(dot(light, normal)) / 3.1415926f;

	float3 ret = 0.f;
	// u32 bm = WaveActiveMax(mat.base_color);
	// if (WaveIsFirstLane()) {
	// 	printf("%d ", bm);
	// }
	if (mat.base_color != 0) {
		Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.base_color)];
		uint w, h, _;
		t.GetDimensions(0, w, h, _);
		f32 x = round(uv.x * w - 0.5f);
	        f32 y = round(uv.y * h - 0.5f);
		ret = t.Load(int3(x, y, 0));
	}

	// RayDesc ray;
	// ray.Origin = position;
	// ray.Direction = light;
	// ray.TMin = 0.1f;
	// ray.TMax = 10000.f;

	// ShadowPayload s;
	// s.unshadowed = false;
	// TraceRay(
	//	ASes[Constants.as.index], 	
	//	RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
	//	0xff, 
	//	0, 
	//	0, 
	//	1, 
	//	ray, 
	//	s
	// );

	payload.value = float4(ret, 1.f);
}

