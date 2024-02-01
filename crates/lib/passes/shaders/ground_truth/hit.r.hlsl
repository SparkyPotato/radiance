#include "common.l.hlsl"

#define PI 3.14159265359f

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

struct LightingInput {
	float4 base_color;
	float3 normal;
	float3 emissive;
	f32 metallic;
	f32 roughness;
};

LightingInput EvalMaterial(Hit hit) {
	Material mat = Constants.materials.load(hit.mat);
	float4 base_color = float4(mat.base_color_factor);
	float3 normal = hit.normal; // TODO: Eval normal map.
	float3 emissive = float3(mat.emissive_factor);
	f32 metallic = mat.metallic_factor;
	f32 roughness = mat.roughness_factor;

	SamplerState s = Constants.sampler.get();
	if (mat.base_color != 0) {
		Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.base_color)];
		base_color *= t.SampleLevel(s, hit.uv, 0.f);
	}
	if (mat.emissive != 0) {
		Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.emissive)];
		emissive *= t.SampleLevel(s, hit.uv, 0.f).xyz;
	}
	if (mat.metallic_roughness != 0) {
		Texture2D t = Texture2Ds[NonUniformResourceIndex(mat.metallic_roughness)];
		float3 mr = t.SampleLevel(s, hit.uv, 0.f).xyz;
		roughness *= mr.g;
		metallic *= mr.b;
	}
	roughness = max(roughness, 0.045f);

	LightingInput l = { base_color, normal, emissive, metallic, roughness };
	return l;
}

f32 D_GGX(f32 n_h, f32 rough) {
	f32 a = n_h * rough;
	f32 k = rough / (1.f - n_h * n_h + a * a);
	return k * k * (1.f / PI);
}

f32 V_GGX(f32 n_v, f32 n_l, f32 rough) {
	f32 a = rough * rough;
	f32 v = n_l * sqrt(n_v * n_v * (1.f - a) + a);
	f32 l = n_v * sqrt(n_l * n_l * (1.f - a) + a);
	return 0.5f / (v + l);
}

float3 F_Schlick(f32 u, float3 f0, f32 f90) {
	return f0 + (float3(f90, f90, f90) - f0) * pow(1.f - u, 5.f);
}

float3 Fd_Burley(f32 n_v, f32 n_l, f32 l_h, f32 rough) {
	f32 f90 = 0.5f + 2.f * rough * l_h * l_h;
	f32 l_f = F_Schlick(n_l, 1.f, f90);
	f32 v_f = F_Schlick(n_v, 1.f, f90);
	return l_f * v_f / PI;
}

float3 EvalBRDF(LightingInput input, float3 v, float3 l) {
	float3 h = normalize(v + l);
	f32 n_v = abs(dot(input.normal, v)) + 1e-5;
	f32 n_l = clamp(dot(input.normal, l), 0.f, 1.f);
	f32 n_h = clamp(dot(input.normal, h), 0.f, 1.f);
	f32 l_h = clamp(dot(l, h), 0.f, 1.f);

	f32 rough = input.roughness * input.roughness;
	float3 f0 = 0.04f * (1.f - input.metallic) + input.base_color.rgb * input.metallic; // TODO: Get from material.
	f32 f90 = 1.f;
	float3 diff_color = (1.f - input.metallic) * input.base_color.rgb;

	f32 D = D_GGX(n_h, rough);
	float3 F = F_Schlick(l_h, f0, f90);
	f32 V = V_GGX(n_v, n_l, rough);

	float3 Fr = D * V * F;
	float3 Fd = diff_color * Fd_Burley(n_v, n_l, l_h, rough);
	return Fr + Fd; // TODO: Sample one at random.
}

[shader("closesthit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attrs) {
	Hit hit = EvalHit(attrs);
	LightingInput input = EvalMaterial(hit);
	float3 v = -WorldRayDirection();
	float3 l = normalize(float3(1.f, 1.f, 1.f));
	float3 c = EvalBRDF(input, v, l);
	float3 ret = c * abs(dot(l, input.normal));

	RayDesc ray;
	ray.Origin = hit.position + hit.normal * 0.0000001f;
	ray.Direction = l;
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

	payload.value = float4(ret * s.unshadowed, 1.f);
}

