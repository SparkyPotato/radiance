#include "radiance-core/util/rng.l.hlsl"

struct LightingData {
	// Normalized vector from the shading point towards the camera.
	float3 view;
	// Normalized vector from the shading point towards the light source.
	float3 light;
	// Normalized halfway between `normal` and `light`.
	float3 half;
	// `dot(normal, view)`
	f32 n_v;
	// `dot(normal, light)`
	f32 n_l;
	// `dot(normal, half)`
	f32 n_h;
	// `dot(light, half)`
	f32 l_h;

	static LightingData calculate(float3 view, float3 light, float3 normal) {
		LightingData ret;
		ret.half = normalize(view + light);
		ret.n_v = saturate(dot(normal, view));
		ret.n_l = saturate(dot(normal, light));
		ret.n_h = saturate(dot(normal, ret.half));
		ret.l_h = saturate(dot(light, ret.half));
		ret.view = view;
		ret.light = light;
		return ret;
	}
};

struct MatInput {
	float4 base_color;
	float3x3 basis;
	float3 emissive;
	f32 metallic;
	// Perceptual roughness squared.
	f32 alpha;

	float3 tangent() {
		return this.basis._m00_m10_m20;
	}

	float3 normal() {
		return this.basis._m01_m11_m21;
	}

	float3 binormal() {
		return this.basis._m02_m12_m22;
	}
};

struct SampleResult {
	float3 color;
	float pdf;
	float3 dir;
};

// Resources:
// PBRT of course: https://pbr-book.org/4ed/contents
// [0] Self Shadow: https://blog.selfshadow.com/links/
// [1] Cook Torrance: https://graphics.pixar.com/library/ReflectanceModel/paper.pdf
// [2] Disney PBR: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] GGX: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// [4] Filament: https://google.github.io/filament/Filament.md.html#materialsystem
// [5] Eric Heitz' Smith extension: https://jcgt.org/published/0003/02/03/paper.pdf 
// [6] Naty Hoffman Notes: https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
// [7] Implenting Disney BRDF: https://schuttejoe.github.io/post/disneybsdf/
// [8] Karis reference: https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

// GGX (Trowbridge-Reitz) normal distribution. [3]
// Basically everyone uses this.
f32 D_GGX(LightingData l, MatInput m) {
	f32 a = l.n_h * m.alpha;
	f32 k = m.alpha / (1.f - l.n_h * l.n_h + a * a);
	return k * k / PI;
}

// Sample the GGX lobe, returning the half vector.
// The PDF is `D * n_h / (4 * l_h)`
float3 Sample_GGX(inout Rng rng, MatInput m) {
	float2 rand = rng.sample2();
	f32 a2 = m.alpha * m.alpha;
	f32 cosH = sqrt(max(0.f, (1.f - rand.x) / ((a2 - 1.f) * rand.x + 1.f)));
	f32 sinH = sqrt(max(0.f, 1.f - cosH * cosH));
	f32 phiH = rand.y * PI * 2.f;
	float3 v = float3(sinH * cos(phiH), cosH, sinH * sin(phiH));
	return mul(m.basis, v);
}

// Height-correlated Smith Visibility Function. [5]
// Where V = G / 4 * n_l * n_v
// Filament says Heitz' height correlated Smith is the correct and exact G term. [4]
// I trust them.
f32 V_GGX(LightingData l, MatInput m) {
	f32 a2 = m.alpha * m.alpha;
	f32 vf = l.n_l * sqrt(l.n_v * l.n_v * (1.f - a2) + a2);
	f32 lf = l.n_v * sqrt(l.n_l * l.n_l * (1.f - a2) + a2);
	// Possible optimization:
	// f32 v = l.n_l * (l.n_v * (1.f - m.alpha) + m.alpha);
	// f32 l = l.n_v * (l.n_l * (1.f - m.alpha) + m.alpha);
	return 0.5f / (vf + lf);
}


// The Schlick fresnel.
float3 F_Schlick(f32 u, float3 f0, float3 f90 = 1.f) {
	return f0 + (f90 - f0) * pow(saturate(1.f - u), 5.f);
}

// The Burley Diffuse BRDF. [2]
float3 BRDF_Burley(LightingData l, MatInput m) {
	f32 f90 = 0.5f + 2.f * m.alpha * l.l_h * l.l_h;
	f32 lf = F_Schlick(l.n_l, 1.f, f90).x;
	f32 vf = F_Schlick(l.n_v, 1.f, f90).x;
	float3 diff_color = (1.f - m.metallic) * m.base_color.xyz;
	return diff_color * lf * vf / PI;
}

// The Cook Torrance Specular BRDF. [1]
// TODO: This is not energy conserving - look at Filament section 4.7. [4]
float3 BRDF_CookTorrance(LightingData l, MatInput m) {
	f32 D = D_GGX(l, m);
	float3 dielectric = 0.04f * (1.f - m.metallic);
	float3 metal = m.base_color.rgb * m.metallic;
	float3 f0 = dielectric + metal; // TODO: Get from material.
	f32 f90 = 1.f;
	float3 F = F_Schlick(l.l_h, f0, f90);
	f32 V = V_GGX(l, m);
	return D * V * F;
}

SampleResult Sample_Burley(inout Rng rng, MatInput m, float3 view) {
	SampleResult ret;
	ret.dir = rng.sample_cos_hemi();
	ret.pdf = ret.dir.y / PI;
	ret.dir = mul(m.basis, ret.dir);
	LightingData l = LightingData::calculate(view, ret.dir, m.normal());
	ret.color = BRDF_Burley(l, m);
	return ret;
}

SampleResult Sample_CookTorrance(inout Rng rng, MatInput m, float3 view) {
	SampleResult ret;
	float3 h = Sample_GGX(rng, m);
	ret.dir = normalize(2.f * dot(view, h) * h - view);
	LightingData l = LightingData::calculate(view, ret.dir, m.normal());
	f32 D = D_GGX(l, m);
	ret.pdf = D * l.n_h / (4.f * l.l_h);
	ret.color = BRDF_CookTorrance(l, m);
	return ret;
}

