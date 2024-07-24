#include "common.l.hlsl"

struct Hit {
	float3 position;
	float3x3 basis;
	float2 uv;
	u32 mat;
};

Hit EvalHit(BuiltInTriangleIntersectionAttributes attrs) {
	Hit hit;

	Instance instance = Constants.instances.load(InstanceID());
	float3x4 smat = ObjectToWorld3x4();
	float4x4 tmat = {
		smat._m00, smat._m01, smat._m02, smat._m03,
		smat._m10, smat._m11, smat._m12, smat._m13,
		smat._m20, smat._m21, smat._m22, smat._m23,
		0.f,       0.f,       0.f,       1.f
	};
	float3 bary = float3(1.0 - attrs.barycentrics.x - attrs.barycentrics.y, attrs.barycentrics.x, attrs.barycentrics.y);
	u32 meshlet_index = GeometryIndex();
	Meshlet meshlet = instance.mesh.load<Meshlet>(0, meshlet_index);

	u32 prim = PrimitiveIndex();
	u32 indices = instance.mesh.load<u32>(meshlet.index_offset, prim);
	uint3 i = uint3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

	Vertex v0 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.x);
	Vertex v1 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.y);
	Vertex v2 = instance.mesh.load<Vertex>(meshlet.vertex_offset, i.z);

	float3 p0 = float3(v0.position);
	float3 p1 = float3(v1.position);
	float3 p2 = float3(v2.position);
	hit.position = mul(tmat, float4(bary.x * p0 + bary.y * p1 + bary.z * p2, 1.f)).xyz;

	float3 n0 = float3(v0.normal);
	float3 n1 = float3(v1.normal);
	float3 n2 = float3(v2.normal);
	float3 normal = normalize(mul(tmat, float4(bary.x * n0 + bary.y * n1 + bary.z * n2, 0.f))).xyz;

	float4 t0 = float4(v0.tangent);
	float4 t1 = float4(v1.tangent);
	float4 t2 = float4(v2.tangent);
	float4 gen = bary.x * t0 + bary.y * t1 + bary.z * t2;
	float4 tangent = float4(normalize(mul(tmat, float4(gen.xyz, 0.f))).xyz, gen.w);

	float3 binormal = cross(normal, tangent.xyz) * tangent.w;

	float3x3 basis = {
		tangent.x, normal.x, binormal.x,
		tangent.y, normal.y, binormal.y,
		tangent.z, normal.z, binormal.z
	};
	hit.basis = basis;

	float2 u0 = float2(v0.uv);
	float2 u1 = float2(v1.uv);
	float2 u2 = float2(v2.uv);
	hit.uv = bary.x * u0 + bary.y * u1 + bary.z * u2;

	return hit;
}

MatInput EvalMaterial(Hit hit) {
	Sampler s = Constants.sampler;
	float2 uv = hit.uv;
	Material mat = Constants.materials.load(hit.mat);
	MatInput ret;
	ret.base_color = float4(mat.base_color_factor);
	float3 normal = float3(0.5f, 1.f, 0.5f);
	normal = normal * 2.f - 1.f;
	normal = normalize(mul(hit.basis, normal));
	hit.basis._m01_m11_m21 = normal; 
	ret.basis = hit.basis;
	ret.emissive = float3(mat.emissive_factor);
	ret.metallic = mat.metallic_factor;
	f32 roughness = max(mat.roughness_factor, 0.045f);
	ret.alpha = roughness * roughness;
	return ret;
}

SampleResult SampleLight(inout Rng rng, MatInput m) {
	SampleResult ret;
	ret.color = 0.5f;
	ret.dir = rng.sample_cos_hemi();
	ret.pdf = ret.dir.y / PI; // TODO: Consolidate with sampling.
	ret.dir = mul(m.basis, ret.dir);
	return ret;
}

float3 EvalBRDF(LightingData l, MatInput m) {
	float3 Fr = BRDF_CookTorrance(l, m);
	float3 Fd = BRDF_Burley(l, m);
	return Fd + Fr;
}

SampleResult SampleBRDF(inout Rng rng, MatInput m, out bool specular) {
	SampleResult ret;
	float3 view = -WorldRayDirection();
	if (rng.sample() < 0.5f) { // TODO: Choose a better way to sample lobes.
		ret = Sample_Burley(rng, m, view);
		specular = false;
	} else {
		ret = Sample_CookTorrance(rng, m, view); 
		specular = true;
	}
	ret.pdf *= 0.5f;
	return ret;
}

struct LightEstimate {
	float3 color;
	float3 dir;
};

LightEstimate EstimateLight(inout Rng rng, MatInput m) {
	SampleResult s = SampleLight(rng, m);
	LightingData l = LightingData::calculate(-WorldRayDirection(), s.dir, m.normal());
	LightEstimate ret;
	ret.color = s.color * abs(dot(s.dir, m.normal())) / s.pdf * EvalBRDF(l, m);
	// TODO: Optimize dot and pdf because it's a cosine sample.
	ret.dir = s.dir;
	return ret;
}

[shader("closesthit")]
void main(inout Payload p, BuiltInTriangleIntersectionAttributes attrs) {
	Rng rng = p.rng;

	Hit hit = EvalHit(attrs);
	MatInput m = EvalMaterial(hit);
	LightEstimate e = EstimateLight(rng, m);

	RayDesc ray;
	ray.Origin = hit.position;
	ray.Direction = e.dir;
	ray.TMin = 0.1f;
	ray.TMax = 10000.f;
	ShadowPayload s;
	s.unshadowed = false;
	TraceRay(
		ASes[Constants.as.index], 	
		RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
		0xff, 0, 0, 1, ray, s
	);

	bool specular;
	SampleResult brdf = SampleBRDF(rng, m, specular);
	p.radiance = e.color * s.unshadowed + m.emissive * p.specular;
	// p.radiance = 0.f;
	p.hit = true;
	p.specular = specular;
	// p.specular = true;
	p.color = brdf.color;
	// p.color = 0.f;
	p.pdf = brdf.pdf;
	// p.pdf = 1.f;
	p.origin = hit.position;
	p.dir = brdf.dir;
	// p.dir = reflect(WorldRayDirection(), m.normal());
	p.dot = abs(dot(p.dir, m.normal()));
	// p.dot = 1.f;
	p.rng = rng;
}

