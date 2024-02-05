#include "common.l.hlsl"
#include "radiance-passes/brdf/brdf.l.hlsl"

struct Hit {
	float3 position;
	float3 normal;
	float2 uv;
	u32 mat;
};

Hit EvalHit(BuiltInTriangleIntersectionAttributes attrs) {
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

	u32 mat = instance.mesh.load<Submesh>(0, meshlet.submesh).mat_index;

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

	Hit hit = { position, normal, uv, mat };
	return hit;
}

MatInput EvalMaterial(Hit hit) {
	Sampler s = Constants.sampler;
	float2 uv = hit.uv;
	Material mat = Constants.materials.load(hit.mat);
	float4 base_color = float4(mat.base_color_factor) * mat.base_color.sample_mip(s, uv, 0.f, 1.f);
	float3 normal = hit.normal; // TODO: Eval normal map.
	float3 emissive = float3(mat.emissive_factor) * mat.emissive.sample_mip(s, uv, 0.f, 1.f).rgb;
	float3 mr = mat.metallic_roughness.sample_mip(s, uv, 0.f, 1.f).rgb;
	f32 metallic = mat.metallic_factor * mr.g;
	f32 roughness = max(mat.roughness_factor * mr.b, 0.045f);

	// SamplerState s = Constants.sampler.get();
	// if (mat.base_color != 0) {
	//	Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.base_color)];
	//	base_color *= t.SampleLevel(s, hit.uv, 0.f);
	// }
	// if (mat.emissive != 0) {
	//	Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.emissive)];
	//	emissive *= t.SampleLevel(s, hit.uv, 0.f).xyz;
	// }
	// if (mat.metallic_roughness != 0) {
	//	Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.metallic_roughness)];
	//	float3 mr = t.SampleLevel(s, hit.uv, 0.f).xyz;
	//	roughness *= mr.g;
	//	metallic *= mr.b;
	// }
	// roughness = max(roughness, 0.045f);

	MatInput ret = { base_color, normal, emissive, metallic, roughness * roughness };
	return ret;
}

float3 EvalBRDF(LightingData l, MatInput m) {
	float3 Fr = BRDF_CookTorrance(l, m);
	float3 Fd = BRDF_Burley(l, m);
	return Fr + Fd; // TODO: Sample one at random.
}

[shader("closesthit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attrs) {
	Hit hit = EvalHit(attrs);
	MatInput m = EvalMaterial(hit);
	float3 vv = -WorldRayDirection();
	float3 lv = m.normal;
	LightingData l = LightingData::calculate(vv, lv, m.normal);

	float3 c = EvalBRDF(l, m);
	float3 irradiance = max(dot(lv, m.normal), 0.f);
	float3 radiance = c * irradiance;

	RayDesc ray;
	ray.Origin = hit.position;
	ray.Direction = lv;
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

	payload.value = float4(radiance, 1.f);
}

