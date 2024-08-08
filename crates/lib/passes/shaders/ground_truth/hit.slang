import graph.types;
import graph.util.rng;
import asset;
import brdf;
import common;

struct Hit {
	float3 position;
	float3x3 basis;
	float2 uv;
	u32 mat;
}

func eval_hit(attrs: BuiltInTriangleIntersectionAttributes) -> Hit {
	Hit hit;

	let instance = Constants.instances[InstanceID()];
	let smat = ObjectToWorld3x4();
	float4x4 tmat = {
		smat._m00, smat._m01, smat._m02, smat._m03,
		smat._m10, smat._m11, smat._m12, smat._m13,
		smat._m20, smat._m21, smat._m22, smat._m23,
		0.f,       0.f,       0.f,       1.f
	};
	let bary = f32x3(1.0 - attrs.barycentrics.x - attrs.barycentrics.y, attrs.barycentrics.x, attrs.barycentrics.y);
	let meshlet_index = GeometryIndex();
	let meshlet = ((Meshlet*)instance.mesh)[meshlet_index];

	let prim = PrimitiveIndex();
	let indices = ((u32*)(instance.mesh + meshlet.index_offset))[prim];
	let i = u32x3(indices >> 0, indices >> 8, indices >> 16) & 0xff;

	let vertices = (Vertex*)(instance.mesh + meshlet.vertex_offset);
	let v0 = vertices[i.x];
	let v1 = vertices[i.y];
	let v2 = vertices[i.z];

	let p0 = f32x3(v0.position);
	let p1 = f32x3(v1.position);
	let p2 = f32x3(v2.position);
	hit.position = mul(tmat, f32x4(bary.x * p0 + bary.y * p1 + bary.z * p2, 1.f)).xyz;

	let n0 = f32x3(v0.normal);
	let n1 = f32x3(v1.normal);
	let n2 = f32x3(v2.normal);
	let normal = normalize(mul(tmat, f32x4(bary.x * n0 + bary.y * n1 + bary.z * n2, 0.f))).xyz;

	// float3x3 basis = {
	//	tangent.x, normal.x, binormal.x,
	//	tangent.y, normal.y, binormal.y,
	//	tangent.z, normal.z, binormal.z
	// };
	hit.basis = 0.f;

	let u0 = f32x2(v0.uv);
	let u1 = f32x2(v1.uv);
	let u2 = f32x2(v2.uv);
	hit.uv = bary.x * u0 + bary.y * u1 + bary.z * u2;

	return hit;
}

func eval_material(hit: Hit) -> MatInput {
	let s = Constants.sampler;
	let uv = hit.uv;
	let mat = Constants.materials[hit.mat];
	MatInput ret;
	ret.base_color = f32x4(mat.base_color_factor);
	var normal = f32x3(0.5f, 1.f, 0.5f);
	normal = normal * 2.f - 1.f;
	normal = normalize(mul(hit.basis, normal));
	hit.basis._m01_m11_m21 = normal; 
	ret.basis = hit.basis;
	ret.emissive = f32x3(mat.emissive_factor);
	ret.metallic = mat.metallic_factor;
	let roughness = max(mat.roughness_factor, 0.045f);
	ret.alpha = roughness * roughness;
	return ret;
}

func sample_light(inout rng: Rng, m: MatInput) -> SampleResult {
	SampleResult ret;
	ret.color = 0.5f;
	ret.dir = rng.sample_cos_hemi();
	ret.pdf = ret.dir.y / PI; // TODO: Consolidate with sampling.
	ret.dir = mul(m.basis, ret.dir);
	return ret;
}

func eval_brdf(l: LightingData, m: MatInput) -> f32x3 {
	let Fr = BRDF_CookTorrance(l, m);
	let Fd = BRDF_Burley(l, m);
	return Fd + Fr;
}

func sample_brdf(inout rng: Rng, m: MatInput, out specular: bool) -> SampleResult {
	SampleResult ret;
	let view = -WorldRayDirection();
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
	f32x3 color;
	f32x3 dir;
}

func estimate_light(inout rng: Rng, m: MatInput) -> LightEstimate {
	let s = sample_light(rng, m);
	let l = LightingData(-WorldRayDirection(), s.dir, m.normal());
	LightEstimate ret;
	ret.color = s.color * abs(dot(s.dir, m.normal())) / s.pdf * eval_brdf(l, m);
	// TODO: Optimize dot and pdf because it's a cosine sample.
	ret.dir = s.dir;
	return ret;
}

[shader("closesthit")]
func main(inout p: Payload, attrs: BuiltInTriangleIntersectionAttributes) {
	var rng = p.rng;

	let hit = eval_hit(attrs);
	let m = eval_material(hit);
	let e = estimate_light(rng, m);

	RayDesc ray;
	ray.Origin = hit.position;
	ray.Direction = e.dir;
	ray.TMin = 0.1f;
	ray.TMax = 10000.f;
	ShadowPayload s;
	s.unshadowed = false;
	Constants.as.trace(
		RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
		0xff, 0, 0, 1, ray, s
	);

	bool specular;
	let brdf = sample_brdf(rng, m, specular);
	p.radiance = e.color * f32(s.unshadowed) + m.emissive * f32(p.specular);
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

